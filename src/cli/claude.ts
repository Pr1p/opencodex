/**
 * `ocx claude [claude args...]` — launch Claude Code wired to the local proxy.
 *
 * Mirrors `ccr code` UX (devlog/260711_claude_inbound/020, 003 E1/E2/E5/G1):
 * ensures the proxy is running, injects the Anthropic env slots, then execs the
 * `claude` CLI with stdio inherited. User-exported env wins except when a stale
 * loopback opencodex base URL points at a different proxy port.
 */
import { spawn } from "node:child_process";
import { loadConfig, resolveEnvValue } from "../config";
import { injectClaudeAgentDefs } from "../claude/agents-inject";
import { effectiveModelEnv, resolveAutoContext } from "../claude/context-windows";
import { refreshGatewayModelCacheFromProxy } from "../claude/gateway-cache";
import { commandInvocation } from "../lib/win-exec";
import { findLiveProxy } from "../server/proxy-liveness";
import type { OcxConfig } from "../types";

export interface ClaudeLaunchEnv {
  [key: string]: string | undefined;
}

export interface ClaudeProfileArgs {
  profile?: string;
  args: string[];
  error?: string;
}

export function parseClaudeProfileArgs(args: string[]): ClaudeProfileArgs {
  const childArgs: string[] = [];
  let profile: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!;
    if (arg === "--profile") {
      const value = args[index + 1]?.trim();
      if (!value || value.startsWith("-")) return { args: childArgs, error: "--profile requires a provider name" };
      if (profile) return { args: childArgs, error: "--profile may only be specified once" };
      profile = value;
      index += 1;
      continue;
    }
    if (arg.startsWith("--profile=")) {
      const value = arg.slice("--profile=".length).trim();
      if (!value) return { args: childArgs, error: "--profile requires a provider name" };
      if (profile) return { args: childArgs, error: "--profile may only be specified once" };
      profile = value;
      continue;
    }
    childArgs.push(arg);
  }
  return { profile, args: childArgs };
}

export function buildClaudeDirectEnv(config: OcxConfig, providerName: string, base: ClaudeLaunchEnv): ClaudeLaunchEnv {
  const provider = config.providers[providerName];
  if (!provider) throw new Error(`Unknown Claude profile "${providerName}". Configure it under the provider settings first.`);
  const direct = provider.claudeDirect;
  if (!direct?.enabled) throw new Error(`Claude direct profile "${providerName}" is not enabled.`);
  if (!direct.baseUrl) throw new Error(`Claude direct profile "${providerName}" has no base URL.`);
  const apiKey = resolveEnvValue(provider.apiKey);
  if (!apiKey) throw new Error(`Claude direct profile "${providerName}" requires the provider API key.`);

  const env: ClaudeLaunchEnv = { ...base };
  env.ANTHROPIC_BASE_URL = direct.baseUrl;
  delete env.ANTHROPIC_AUTH_TOKEN;
  delete env.ANTHROPIC_API_KEY;
  if (direct.authMode === "api-key") env.ANTHROPIC_API_KEY = apiKey;
  else env.ANTHROPIC_AUTH_TOKEN = apiKey;
  if (direct.model) env.ANTHROPIC_MODEL = direct.model;
  env.CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST = "1";
  delete env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY;
  return env;
}

export function buildClaudeNativeEnv(base: ClaudeLaunchEnv): ClaudeLaunchEnv {
  const env: ClaudeLaunchEnv = { ...base };
  if (env.ANTHROPIC_AUTH_TOKEN === "opencodex-proxy") delete env.ANTHROPIC_AUTH_TOKEN;
  if (env.ANTHROPIC_BASE_URL) {
    try {
      const parsed = new URL(env.ANTHROPIC_BASE_URL);
      if (parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1") delete env.ANTHROPIC_BASE_URL;
    } catch { /* preserve an explicit non-URL value */ }
  }
  delete env.CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST;
  delete env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY;
  return env;
}

/**
 * Pure env assembly (unit-tested): never sets ANTHROPIC_API_KEY (setting both
 * token vars triggers Claude Code's auth-conflict warning, 003 E1), and never
 * overrides variables the user already exported, apart from stale loopback
 * ANTHROPIC_BASE_URL values owned by a previous opencodex launch.
 */
export function buildClaudeEnv(config: OcxConfig, port: number, base: ClaudeLaunchEnv, contextWindows: Record<string, number> = {}): ClaudeLaunchEnv {
  const env: ClaudeLaunchEnv = { ...base };
  const setDefault = (name: string, value: string | undefined) => {
    if (value === undefined || value.length === 0) return;
    if (env[name] !== undefined && env[name] !== "") return; // user wins
    env[name] = value;
  };
  setDefault("ANTHROPIC_BASE_URL", `http://127.0.0.1:${port}`);
  const existingBaseUrl = env.ANTHROPIC_BASE_URL;
  if (existingBaseUrl) {
    try {
      const parsed = new URL(existingBaseUrl);
      const isLoopback = parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1";
      if (isLoopback && parsed.port !== "" && Number(parsed.port) !== port) {
        const replacement = `http://127.0.0.1:${port}`;
        console.error(`⚠ Replacing stale opencodex ANTHROPIC_BASE_URL ${existingBaseUrl} with ${replacement}.`);
        env.ANTHROPIC_BASE_URL = replacement;
      }
    } catch {
      // Preserve user-provided values that are not parseable URLs.
    }
  }
  // Subscription-preserving default (teamclaude --no-mitm / Vercel gateway pattern):
  // setting ANTHROPIC_AUTH_TOKEN/API_KEY disables claude.ai connectors and overrides
  // the user's Claude login. Only inject a token when the proxy actually requires an
  // admission key; otherwise Claude Code keeps its own OAuth and sends it to us —
  // native claude models then pass through verbatim (see server/claude-messages.ts).
  if ((config.apiKeys?.length ?? 0) > 0) {
    setDefault("ANTHROPIC_AUTH_TOKEN", config.apiKeys![0].key);
  }
  if (!env.ANTHROPIC_AUTH_TOKEN && config.claudeCode?.authMode === "proxy") {
    env.ANTHROPIC_AUTH_TOKEN = "opencodex-proxy";
  }
  // NOTE: do NOT set _CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL here. While it enables
  // Design/Remote Control, it DISABLES gateway model discovery (Claude Code's eligibility
  // check returns false when isFirstPartyBaseUrl() is true). Model routing through the
  // proxy is essential; Design/Remote Control are secondary features.
  // Connectors still work because they check OAuth state ($o()), not base URL (Gd()).
  // Native /model picker discovery ("From gateway", Claude Code >= 2.1.129).
  setDefault("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1");
  // Host-managed routing guard (devlog 260720_claude_authmode_persist/020): with
  // this flag in the spawn env, Claude Code strips provider-managed vars
  // (ANTHROPIC_BASE_URL/AUTH_TOKEN/API_KEY, model slots) from settings-sourced
  // env (managedEnv.ts), so a leftover cc-switch/CCR ~/.claude/settings.json
  // env block cannot silently hijack proxy routing away from opencodex.
  // setDefault: an explicit user export (e.g. =0, isEnvTruthy-false) still wins.
  // Intentional contract change: settings.env model slots are also stripped in
  // ocx claude runs — use the top-level settings "model" field or opt out.
  setDefault("CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST", "1");
  // Opt-in effort forcing (devlog 136 B6): opus-shaped aliases already carry
  // output_config.effort, so this is OFF unless the user enables it in config.
  if (config.claudeCode?.alwaysEnableEffort === true) {
    setDefault("CLAUDE_CODE_ALWAYS_ENABLE_EFFORT", "1");
  }
  // Context-window override: the official pair — MAX_CONTEXT_TOKENS alone is ignored
  // for recognized claude-shaped ids unless DISABLE_COMPACT=1 rides along (devlog 135).
  const maxCtx = config.claudeCode?.maxContextTokens;
  if (typeof maxCtx === "number" && Number.isFinite(maxCtx) && maxCtx > 0) {
    setDefault("CLAUDE_CODE_MAX_CONTEXT_TOKENS", String(Math.floor(maxCtx)));
    setDefault("DISABLE_COMPACT", "1");
  }
  // Auto-context (devlog 260712 020): min(believed window, env) inside the CLI means
  // one global env acts as a per-model floor — [1m]-marked models compact here while
  // unmarked (200k-accounted) models keep their default behavior. Inert when the
  // legacy maxContextTokens pair above is set (resolveAutoContext handles that).
  // A user-exported value drives the marking predicate too (audit 021 #2) so the
  // [1m] marker and the compaction threshold can never separate.
  const userAutoCompact = typeof base.CLAUDE_CODE_AUTO_COMPACT_WINDOW === "string" && base.CLAUDE_CODE_AUTO_COMPACT_WINDOW !== ""
    ? base.CLAUDE_CODE_AUTO_COMPACT_WINDOW
    : undefined;
  const auto = resolveAutoContext(config.claudeCode, userAutoCompact);
  if (auto.enabled) {
    setDefault("CLAUDE_CODE_AUTO_COMPACT_WINDOW", String(auto.compactWindow));
  }
  // Model slots (devlog 260712 B2): default + four tier defaults + legacy small-fast,
  // with automatic [1m] context-variant marking when the slot's target model has an
  // authoritative >=1M window (Claude Code then accounts 1M, compaction preserved).
  for (const [name, value] of Object.entries(effectiveModelEnv(config.claudeCode, contextWindows, auto))) {
    setDefault(name, value);
  }
  return env;
}

/**
 * Context-window map from the RUNNING proxy's management API (warm TTL cache; the
 * daemon registers every selector form — audit R3#1). 3s bound + auth header
 * (OPENCODEX_API_AUTH_TOKEN first, config key fallback — audit R4#1). Failure → {}
 * (no [1m] marking, conservative).
 */
export async function fetchClaudeContextWindows(config: OcxConfig, port: number, timeoutMs = 3_000): Promise<Record<string, number>> {
  try {
    const headers = new Headers();
    const token = process.env.OPENCODEX_API_AUTH_TOKEN || config.apiKeys?.[0]?.key;
    if (token) headers.set("x-opencodex-api-key", token);
    const res = await fetch(`http://127.0.0.1:${port}/api/claude-code`, {
      headers,
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return {};
    const body = await res.json() as { contextWindows?: Record<string, number> };
    return body.contextWindows && typeof body.contextWindows === "object" ? body.contextWindows : {};
  } catch {
    console.error("⚠ 모델 컨텍스트 정보를 불러오지 못했습니다 — 1M 자동 표시는 이번 실행에서 생략됩니다.");
    return {};
  }
}

async function ensureProxyForClaude(): Promise<number | null> {
  const live = await findLiveProxy();
  if (live) return live.port;
  const cfgPort = loadConfig().port;
  const pinPort = typeof cfgPort === "number" && cfgPort > 0 ? cfgPort : 10100;
  const child = spawn(process.execPath, [process.argv[1], "start", "--port", String(pinPort)], {
    detached: true,
    stdio: "ignore",
    windowsHide: true,
    env: { ...process.env, OCX_SERVICE: "1" },
  });
  child.unref();
  const deadline = Date.now() + 8_000;
  while (Date.now() < deadline) {
    const started = await findLiveProxy();
    if (started) return started.port;
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  return null;
}

const CLAUDE_INSTALL_HINT = "❌ `claude` CLI not found. Install it first: npm install -g @anthropic-ai/claude-code";

/**
 * cmd.exe reports command-not-found as exit 9009 (the win32 launcher routes `.cmd`
 * shims through cmd.exe, so ENOENT never fires there). Signal exits are not hints.
 * Devlog 260715_cross_platform_audit/020.
 */
export function claudeNotFoundHint(
  code: number | null,
  signal: NodeJS.Signals | null,
  platform: NodeJS.Platform = process.platform,
): string | null {
  return platform === "win32" && code === 9009 && !signal ? CLAUDE_INSTALL_HINT : null;
}

async function spawnClaude(args: string[], env: ClaudeLaunchEnv): Promise<number> {
  return await new Promise<number>(resolve => {
    const inv = commandInvocation("claude", args);
    const child = spawn(inv.file, inv.args, { stdio: "inherit", env: env as NodeJS.ProcessEnv, ...inv.options });
    child.on("error", (err: NodeJS.ErrnoException) => {
      if (err.code === "ENOENT") console.error(CLAUDE_INSTALL_HINT);
      else console.error(`Failed to launch claude: ${err.message}`);
      resolve(1);
    });
    child.on("exit", (code, signal) => {
      const hint = claudeNotFoundHint(code, signal);
      if (hint) console.error(hint);
      resolve(signal ? 1 : code ?? 0);
    });
  });
}

export async function cmdClaude(args: string[]): Promise<number> {
  const parsedArgs = parseClaudeProfileArgs(args);
  if (parsedArgs.error) {
    console.error(parsedArgs.error);
    return 1;
  }
  const config = loadConfig();
  if (config.claudeCode?.enabled === false) {
    console.error("Claude inbound is disabled (config.claudeCode.enabled=false — flip the Claude ON toggle in the GUI or edit config).");
    return 1;
  }
  if (parsedArgs.profile === "native" || parsedArgs.profile === "subscription") {
    return await spawnClaude(parsedArgs.args, buildClaudeNativeEnv(process.env));
  }
  if (parsedArgs.profile) {
    try {
      return await spawnClaude(parsedArgs.args, buildClaudeDirectEnv(config, parsedArgs.profile, process.env));
    } catch (error) {
      console.error(error instanceof Error ? error.message : String(error));
      return 1;
    }
  }
  const port = await ensureProxyForClaude();
  if (!port) {
    console.error("❌ Proxy did not become healthy after starting.");
    return 1;
  }
  const contextWindows = await fetchClaudeContextWindows(config, port);
  const env = buildClaudeEnv(config, port, process.env, contextWindows);
  // Pre-write the CLI's gateway-model cache (devlog 030): without a token the CLI
  // never refreshes it, so the picker would keep showing yesterday's aliases.
  try {
    const cachePath = await refreshGatewayModelCacheFromProxy(port);
    if (cachePath === null) {
      console.error("⚠ Gateway model cache could not be refreshed; the model picker may be stale.");
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`⚠ Gateway model cache could not be refreshed: ${message}`);
  }
  // Sync roster agents (devlog 070): subagentModels + self -> ~/.claude/agents/ocx-*.md.
  try {
    const written = injectClaudeAgentDefs(config, contextWindows);
    if (written === null) {
      console.error("⚠ Claude agent definitions could not be synced; check ~/.claude/agents permissions.");
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`⚠ Claude agent definitions could not be synced: ${message}`);
  }
  return await new Promise<number>(resolve => {
    const inv = commandInvocation("claude", parsedArgs.args);
    const child = spawn(inv.file, inv.args, { stdio: "inherit", env: env as NodeJS.ProcessEnv, ...inv.options });
    child.on("error", (err: NodeJS.ErrnoException) => {
      if (err.code === "ENOENT") {
        console.error(CLAUDE_INSTALL_HINT);
      } else {
        console.error(`❌ Failed to launch claude: ${err.message}`);
      }
      resolve(1);
    });
    child.on("exit", (code, signal) => {
      const hint = claudeNotFoundHint(code, signal);
      if (hint) console.error(hint);
      resolve(signal ? 1 : code ?? 0);
    });
  });
}

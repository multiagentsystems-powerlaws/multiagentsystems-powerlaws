"""
src/tools/tools.py

Benchmark-native tool registry.

Web search: uses duckduckgo_search (completely free, no API key).
  Install: pip install duckduckgo-search
  Falls back to googlesearch-python if ddg unavailable.
  Last resort: requests + DuckDuckGo HTML scrape.

All other tools: no external API keys required.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── Result dataclass ──────────────────────────────────────────

@dataclass
class ToolResult:
    tool_name:  str
    tool_input: str
    output:     str
    success:    bool
    error:      Optional[str] = None
    latency_ms: float = 0.0

    def summary(self, max_chars: int = 500) -> str:
        if not self.output:
            return self.error or "(no output)"
        return self.output[:max_chars] + ("…" if len(self.output) > max_chars else "")


# ── Base tool ─────────────────────────────────────────────────

class BaseTool:
    name: str = "base_tool"
    description: str = ""

    def run(self, tool_input: str, **kwargs) -> ToolResult:
        raise NotImplementedError


# ── GAIA tools ────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Free web search — no API key required.

    Priority order:
      1. duckduckgo-search  (pip install duckduckgo-search)
      2. googlesearch-python (pip install googlesearch-python)
      3. requests + DDG HTML fallback (always works, no install)
    """
    name = "web_search"
    description = (
        "Search the web for factual information. "
        "Input: a search query string. "
        "Output: top search results as plain text."
    )
    MAX_RESULTS = 5

    def run(self, tool_input: str, **kwargs) -> ToolResult:
        t0 = time.time()
        query = tool_input.strip()

        # ── Strategy 1: duckduckgo-search ─────────────────────
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.MAX_RESULTS))
            if results:
                lines = []
                for r in results:
                    lines.append(f"[{r.get('title','')}]\n{r.get('href','')}\n{r.get('body','')}")
                output = "\n\n".join(lines)
                return ToolResult(
                    tool_name=self.name, tool_input=query,
                    output=output[:3000], success=True,
                    latency_ms=(time.time() - t0) * 1000,
                )
        except ImportError:
            pass
        except Exception as e:
            pass  # fall through to next strategy

        # ── Strategy 2: googlesearch-python ───────────────────
        try:
            from googlesearch import search
            urls = list(search(query, num_results=self.MAX_RESULTS, sleep_interval=1))
            if urls:
                output = "\n".join(urls)
                return ToolResult(
                    tool_name=self.name, tool_input=query,
                    output=output, success=True,
                    latency_ms=(time.time() - t0) * 1000,
                )
        except ImportError:
            pass
        except Exception:
            pass

        # ── Strategy 3: requests + DDG HTML (no install needed) ─
        try:
            import urllib.parse, re
            import requests
            headers = {"User-Agent": "Mozilla/5.0"}
            url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
            resp = requests.get(url, headers=headers, timeout=10)
            # Extract result snippets from HTML
            snippets = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL
            )
            clean = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:self.MAX_RESULTS]]
            if clean:
                output = "\n\n".join(clean)
                return ToolResult(
                    tool_name=self.name, tool_input=query,
                    output=output[:3000], success=True,
                    latency_ms=(time.time() - t0) * 1000,
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, tool_input=query,
                output="", success=False,
                error=f"All search strategies failed. Last error: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )

        return ToolResult(
            tool_name=self.name, tool_input=query,
            output="", success=False,
            error="No search results returned by any strategy.",
            latency_ms=(time.time() - t0) * 1000,
        )


class PythonExecTool(BaseTool):
    """Execute Python code in a subprocess. Timeout: 30s."""
    name = "python_exec"
    description = (
        "Execute Python code and return the output. "
        "Input: valid Python code as a string. Output: stdout and stderr."
    )
    TIMEOUT = 30

    def run(self, tool_input: str, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(tool_input)
                fname = f.name
            result = subprocess.run(
                ["python3", fname], capture_output=True, text=True,
                timeout=self.TIMEOUT,
            )
            output = result.stdout + result.stderr
            return ToolResult(
                tool_name=self.name, tool_input=tool_input[:200],
                output=output[:2000], success=result.returncode == 0,
                error=None if result.returncode == 0 else result.stderr[:500],
                latency_ms=(time.time() - t0) * 1000,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name, tool_input=tool_input[:200],
                output="", success=False, error=f"Timeout after {self.TIMEOUT}s",
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, tool_input=tool_input[:200],
                output="", success=False, error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )


class FileReaderTool(BaseTool):
    """Read a file by path. Output truncated to 4000 chars."""
    name = "file_reader"
    description = (
        "Read the contents of a file. "
        "Input: file path. Output: file contents as plain text."
    )
    MAX_CHARS = 4000

    def run(self, tool_input: str, **kwargs) -> ToolResult:
        t0 = time.time()
        path = tool_input.strip()
        try:
            with open(path) as f:
                content = f.read(self.MAX_CHARS)
            return ToolResult(
                tool_name=self.name, tool_input=path,
                output=content + (" [truncated]" if len(content) == self.MAX_CHARS else ""),
                success=True, latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, tool_input=path,
                output="", success=False, error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )


# ── SWE-bench tool ────────────────────────────────────────────

class BashTool(BaseTool):
    """Execute bash commands. Timeout: 60s."""
    name = "bash"
    description = (
        "Execute a bash command and return stdout + stderr. "
        "Supports: grep, find, cat, git, python, pytest, etc."
    )
    TIMEOUT = 60

    def run(self, tool_input: str, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            result = subprocess.run(
                tool_input, shell=True, capture_output=True,
                text=True, timeout=self.TIMEOUT,
            )
            output = (result.stdout + result.stderr)[:3000]
            return ToolResult(
                tool_name=self.name, tool_input=tool_input[:300],
                output=output, success=result.returncode == 0,
                error=None if result.returncode == 0 else result.stderr[:500],
                latency_ms=(time.time() - t0) * 1000,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name, tool_input=tool_input[:300],
                output="", success=False, error=f"Timeout after {self.TIMEOUT}s",
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, tool_input=tool_input[:300],
                output="", success=False, error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )


# ── REALM-Bench tool ──────────────────────────────────────────

class CalculatorTool(BaseTool):
    """Safe arithmetic evaluator. No external dependencies."""
    name = "calculator"
    description = (
        "Evaluate a mathematical expression. "
        "Input: arithmetic expression, e.g. '(12 * 8) / 3 + 7'. Output: result."
    )

    def run(self, tool_input: str, **kwargs) -> ToolResult:
        t0 = time.time()
        expr = tool_input.strip()
        try:
            import math
            safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            safe.update({"abs": abs, "round": round})
            result = eval(expr, {"__builtins__": {}}, safe)  # noqa: S307
            return ToolResult(
                tool_name=self.name, tool_input=expr,
                output=str(result), success=True,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, tool_input=expr,
                output="", success=False, error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )


# ── MultiAgentBench tool ──────────────────────────────────────

class EnvironmentActionTool(BaseTool):
    """Interface to MultiAgentBench environment. Pass environment= in kwargs."""
    name = "environment_action"
    description = (
        "Execute an action in the current task environment. "
        "Input: action string. Output: environment response."
    )

    def run(self, tool_input: str, **kwargs) -> ToolResult:
        t0 = time.time()
        environment = kwargs.get("environment")
        if environment is None:
            return ToolResult(
                tool_name=self.name, tool_input=tool_input,
                output="", success=False,
                error="No environment instance provided.",
                latency_ms=(time.time() - t0) * 1000,
            )
        try:
            response = environment.step(tool_input)
            return ToolResult(
                tool_name=self.name, tool_input=tool_input[:300],
                output=str(response)[:2000], success=True,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, tool_input=tool_input[:300],
                output="", success=False, error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )


# ── Registry ──────────────────────────────────────────────────

_ALL_TOOLS: Dict[str, BaseTool] = {
    "web_search":          WebSearchTool(),
    "python_exec":         PythonExecTool(),
    "file_reader":         FileReaderTool(),
    "bash":                BashTool(),
    "calculator":          CalculatorTool(),
    "environment_action":  EnvironmentActionTool(),
}

BENCHMARK_TOOLS: Dict[str, List[str]] = {
    "gaia":            ["web_search", "python_exec", "file_reader"],
    "swebench":        ["bash"],
    "realmb":          ["calculator"],
    "multiagentbench": ["environment_action"],
}


def get_tools_for_benchmark(benchmark: str) -> List[BaseTool]:
    benchmark = benchmark.lower()
    if benchmark not in BENCHMARK_TOOLS:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Available: {list(BENCHMARK_TOOLS.keys())}"
        )
    return [_ALL_TOOLS[name] for name in BENCHMARK_TOOLS[benchmark]]


def get_tool_names_for_benchmark(benchmark: str) -> List[str]:
    return [t.name for t in get_tools_for_benchmark(benchmark)]


def run_tool(
    tool_name:   str,
    tool_input:  str,
    benchmark:   str,
    environment: Any = None,
) -> ToolResult:
    allowed = get_tool_names_for_benchmark(benchmark)
    if tool_name not in allowed:
        return ToolResult(
            tool_name=tool_name, tool_input=tool_input,
            output="", success=False,
            error=f"Tool '{tool_name}' not allowed for benchmark '{benchmark}'. "
                  f"Allowed: {allowed}",
        )
    return _ALL_TOOLS[tool_name].run(tool_input, environment=environment)
with open("tools.py") as f:
    text = f.read()

# Find and replace just the WebSearchTool.run() body
old = '            result = f"[web_search stub] Query: {tool_input}\\nResults would appear here."\n            return ToolResult(\n                tool_name=self.name,\n                tool_input=tool_input,\n                output=result,\n                success=True,\n                latency_ms=(time.time() - t0) * 1000,\n            )\n        except Exception as e:\n            return ToolResult(\n                tool_name=self.name,\n                tool_input=tool_input,\n                output="",\n                success=False,\n                error=str(e),\n                latency_ms=(time.time() - t0) * 1000,\n            )'

new = '''            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(tool_input.strip(), max_results=5))
            if not results:
                raise ValueError("No results")
            lines = [
                f"[{r.get('title','')}]\\n{r.get('href','')}\\n{r.get('body','')}"
                for r in results
            ]
            return ToolResult(
                tool_name=self.name, tool_input=tool_input,
                output="\\n\\n".join(lines)[:3000], success=True,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            # Fallback: requests + DDG HTML
            try:
                import urllib.parse, re, requests
                url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(tool_input)
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                snippets = re.findall(r\'class="result__snippet"[^>]*>(.*?)</a>\', resp.text, re.DOTALL)
                clean = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:5]]
                return ToolResult(
                    tool_name=self.name, tool_input=tool_input,
                    output="\\n\\n".join(clean)[:3000] if clean else "",
                    success=bool(clean),
                    error=None if clean else str(e),
                    latency_ms=(time.time() - t0) * 1000,
                )
            except Exception as e2:
                return ToolResult(
                    tool_name=self.name, tool_input=tool_input,
                    output="", success=False, error=str(e2),
                    latency_ms=(time.time() - t0) * 1000,
                )'''

if old in text:
    text = text.replace(old, new, 1)
    with open("tools.py", "w") as f:
        f.write(text)
    print("SUCCESS")
else:
    # Show what's actually there so we can match it
    idx = text.find("web_search stub")
    if idx != -1:
        print("FOUND STUB AT:", idx)
        print(repr(text[idx-50:idx+300]))
    else:
        idx = text.find("Stub implementation")
        print("FOUND ALT AT:", idx)
        print(repr(text[max(0,idx-200):idx+400]))
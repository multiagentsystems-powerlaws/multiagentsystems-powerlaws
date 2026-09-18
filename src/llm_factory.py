"""Shared LLM client factory: dispatches a model name string to the right
LangChain chat client (OpenAI vs. Anthropic) based on substring matching.
"""


def build_llm(model_name: str, temperature: float = 0.7):
    name = model_name.lower()
    if any(x in name for x in ("gpt", "o1", "o3", "o4")):
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(model=model_name, temperature=temperature)
    if "claude" in name:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
            return ChatAnthropic(model=model_name, temperature=temperature)
        except ImportError:
            raise ImportError("pip install langchain-anthropic")
    from langchain_openai import ChatOpenAI  # type: ignore
    return ChatOpenAI(model=model_name, temperature=temperature)

"""Settings: optional AI provider (off by default) and privacy information."""

from __future__ import annotations

import streamlit as st

from torusplot.ai import (
    PROVIDERS,
    RECOMMENDED_OLLAMA_MODELS,
    AISettings,
    ProviderError,
    get_provider,
    load_settings,
)

PROVIDER_LABELS = {
    "mock": "Demo (offline, canned answers; for trying the screens)",
    "ollama": "Ollama (free, runs on this computer)",
    "gemini": "Google Gemini (your own API key)",
    "anthropic": "Anthropic (your own API key)",
    "openai": "OpenAI (your own API key)",
}
KEY_FIELDS = {"gemini": "gemini_api_key", "anthropic": "anthropic_api_key", "openai": "openai_api_key"}
KEY_ENV = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def ai_settings() -> AISettings:
    """The session's AI settings (start from environment variables / the .env file)."""
    if "ai_settings" not in st.session_state:
        try:
            st.session_state["ai_settings"] = load_settings()
        except Exception:  # noqa: BLE001 - a malformed .env must not break the app
            st.session_state["ai_settings"] = AISettings()
    return st.session_state["ai_settings"]


def render() -> None:
    st.header("Settings")
    settings = ai_settings()
    st.subheader("AI features (optional)")
    st.caption(
        "TorusPlot works fully without AI. When enabled, AI can only propose how a figure looks; "
        "it can never compute or change statistics."
    )
    enabled = st.toggle("Enable AI features", value=settings.enabled, key="ai_enabled",
                        help="Off by default. Nothing is sent anywhere while this is off.")
    providers = list(PROVIDERS)
    provider = st.selectbox("Provider", providers, index=providers.index(settings.provider) if settings.provider in providers else 0,
                            format_func=lambda p: PROVIDER_LABELS.get(p, p), key="ai_provider")
    update: dict = {"enabled": bool(enabled), "provider": provider}

    if provider == "ollama":
        st.markdown("**Ollama** is a free program that runs AI models on your own computer: "
                    "install it from https://ollama.com, then download a model with the command shown below.")
        update["ollama_host"] = st.text_input("Ollama address", value=settings.ollama_host, key="ai_ollama_host")
        update["ollama_text_model"] = st.text_input("Text model", value=settings.ollama_text_model, key="ai_ollama_text",
                                                    help="Used for 'Describe it' and outside instructions.")
        update["ollama_vision_model"] = st.text_input("Vision model", value=settings.ollama_vision_model, key="ai_ollama_vision",
                                                      help="Used for 'Match this figure' (reads images).")
        st.markdown("Recommended models (Apache-2.0 / MIT licensed only; TorusPlot never bundles or downloads models):")
        st.dataframe(
            [{"Model": m["tag"], "Use": m["role"], "License": m["license"], "Get it": m["pull"], "License source": m["source"]}
             for m in RECOMMENDED_OLLAMA_MODELS], hide_index=True, width="stretch")
        st.caption("Check a model's license yourself with:  ollama show <model> --license")
    elif provider in KEY_FIELDS:
        field = f"{provider}_model"
        update[field] = st.text_input(
            "Model name", value=getattr(settings, field), key=f"ai_{provider}_model",
            help="The exact model name from your provider account." + (
                "" if provider == "anthropic" else " There is no built-in default because model names change."),
        )
        has_key = settings.has_key(provider)
        st.markdown(f"API key: **{'found' if has_key else 'not set'}** "
                    f"(read from `{KEY_ENV[provider]}` in the `.env` file next to the app; see `.env.example`).")
        typed = st.text_input("Or paste a key for this session only", type="password", key=f"ai_{provider}_key",
                              help="Kept in memory until you close the app. It is never written to disk or shown again.")
        if typed.strip():
            update[KEY_FIELDS[provider]] = typed.strip()
        if provider == "anthropic":
            st.caption("The Anthropic provider needs one extra free package: `pip install -r requirements-ai.txt`. "
                       "Requests opt in to the provider's automatic safety fallback to another of its models.")
        st.warning(
            "With a cloud provider, the text you type into AI boxes and the images you upload there are sent to that "
            "company under **your** account. Its terms, pricing and data policies apply. Your data values are not "
            "included in AI prompts; table and column names are."
        )
    else:
        st.info("Demo mode returns canned example answers so you can try the AI screens without any setup.")

    try:
        new = settings.model_copy(update=update)
        new = AISettings.model_validate(new.model_dump() | {k: v for k, v in update.items()})
    except Exception as exc:  # noqa: BLE001
        st.error(f"Invalid setting: {exc}")
        new = settings
    st.session_state["ai_settings"] = new

    if st.button("Test the provider", key="ai_test"):
        try:
            ok, message = get_provider(new, allow_disabled=True).available()
        except ProviderError as exc:
            ok, message = False, str(exc)
        (st.success if ok else st.warning)(new.redact(message) if hasattr(new, "redact") else message)
    with st.expander("Current AI settings (keys are never shown)"):
        st.json(new.public_dict())

    st.subheader("Privacy")
    st.markdown(
        "- TorusPlot runs on this computer. Your data, figures and results are not uploaded anywhere.\n"
        "- The app does not collect usage statistics.\n"
        "- The only time anything leaves this computer is when you enable a **cloud** AI provider and use an AI box."
    )

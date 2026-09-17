"""
functions/ensure_language.py

Ensures the model's response ends up in the same language as the user's
message. This is needed because fine-tuning is done only in English (a
decision made for the Virtual-Therapist project) - every other language,
including Portuguese, depends on this translation layer, not on the model
itself.

Unlike a "blind" translation (always translate), this detects the actual
language of the response and only translates when it doesn't match the
user's language - avoiding unnecessary translation calls when the model
already answered in the right language (which can happen with some
multilingual base models).

Typical usage inside the main conversation loop:

    from functions.ensure_language import ensure_response_language

    response = fine_tuned_model.generate(...)   # generated in English
    response = ensure_response_language(response, user_input)
"""
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator


def ensure_response_language(response_text: str, user_message: str) -> str:
    """
    Translates response_text into the language of user_message, if they differ.
    Never raises an exception outward - on failure (detection or translation),
    returns the original text instead of interrupting the conversation.
    """
    try:
        user_lang = detect(user_message)
    except LangDetectException:
        return response_text  # couldn't detect the user's language

    try:
        response_lang = detect(response_text)
    except LangDetectException:
        response_lang = None

    if response_lang == user_lang:
        return response_text  # already in the right language

    try:
        translated = GoogleTranslator(source="auto", target=user_lang).translate(response_text)
        return translated or response_text
    except Exception as e:
        print(f"[TRANSLATION WARNING] Failed to translate response into '{user_lang}': {e}")
        return response_text  # translation failed -> return the original, never block the conversation


if __name__ == "__main__":
    # Note: these tests need real internet access (a call to Google Translate).
    # They won't run inside network-restricted environments.
    llm_response = "I understand this must be really hard for you."
    user_message = "Sinto-me muito em baixo hoje."  # kept in Portuguese: this is the input being tested
    print(ensure_response_language(llm_response, user_message))

    llm_response_2 = "Percebo que isto deve ser muito difícil para ti."  # already correct, should not be translated
    print(ensure_response_language(llm_response_2, user_message))

from rag_assistant.text import content_words, split_sentences


def test_headings_and_sentences_are_separate_units():
    text = "# Title\n\nFirst sentence. Second one!  Third?\nLast line"
    assert split_sentences(text) == [
        "# Title",
        "First sentence.",
        "Second one!",
        "Third?",
        "Last line",
    ]


def test_hard_wrapped_lines_are_joined():
    text = (
        "Tools are functions the agent may call, such as\nsearching the web. Each tool\nhas a name."
    )
    assert split_sentences(text) == [
        "Tools are functions the agent may call, such as searching the web.",
        "Each tool has a name.",
    ]


def test_list_items_stay_separate():
    text = "Steps:\n- install it\n- run it\n  with flags\n1. numbered"
    assert split_sentences(text) == ["Steps:", "- install it", "- run it with flags", "1. numbered"]


def test_heading_directly_followed_by_text():
    assert split_sentences("## Tools\nTools are functions.") == ["## Tools", "Tools are functions."]


def test_abbreviations_do_not_end_sentences():
    text = "Use a vector store, e.g. Qdrant. It is fast."
    assert split_sentences(text) == ["Use a vector store, e.g. Qdrant.", "It is fast."]


def test_numbers_at_the_end_of_a_sentence_still_end_it():
    assert split_sentences("The limit is 1000. Then it stops.") == [
        "The limit is 1000.",
        "Then it stops.",
    ]


def test_content_words_keep_letters_from_any_script():
    words = content_words("What is the café crème? Данные, chunk_size, 2024.")
    assert words == ["café", "crème", "данные", "chunk", "size", "2024"]

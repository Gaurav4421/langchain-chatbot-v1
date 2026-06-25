"""
============================================================
  Production-Quality Terminal AI Chatbot
  Stack: Python | LangChain | HuggingFace | TinyLlama
============================================================

Architecture:
  User Input
      │
      ▼
  ChatPromptTemplate (System + History + Human)
      │
      ▼
  ChatHuggingFace (chat-aware model wrapper)
      │
      ▼
  HuggingFacePipeline (raw Transformers pipeline)
      │
      ▼
  TinyLlama-1.1B-Chat-v1.0 (local model inference)
      │
      ▼
  AIMessage response → printed to terminal
      │
      ▼
  ChatMessageHistory (append both turns)
      │
      └──────────────────────────► (next iteration)
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
# Each import is explained in detail below.

# langchain_huggingface provides the bridge between HuggingFace
# Transformers and the LangChain ecosystem.
#
# HuggingFacePipeline: Wraps a native HuggingFace `pipeline` object
#   into a LangChain-compatible LLM. It exposes .invoke() so LangChain
#   can call it like any other LLM. At this stage, it only handles
#   raw strings (not structured chat messages).
#
# ChatHuggingFace: A second wrapper around HuggingFacePipeline that
#   adds chat message awareness. It converts lists of LangChain message
#   objects (SystemMessage, HumanMessage, AIMessage) into the exact
#   prompt format the underlying model expects (e.g., TinyLlama's
#   <|system|>...<|user|>...<|assistant|> format), using the model's
#   built-in Jinja2 chat template.
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# transformers.pipeline is HuggingFace's high-level API for loading and
# running models. The `pipeline` function handles:
#   - downloading model weights from HuggingFace Hub (or loading from cache)
#   - setting up the tokenizer
#   - running inference end-to-end (text in → text out)
# We use task="text-generation" which tells the pipeline this is an
# autoregressive generation model (i.e., it predicts the next token).
from transformers import pipeline

# ChatPromptTemplate: A structured prompt builder that accepts a list of
#   message templates. It can contain fixed messages (like SystemMessage)
#   and dynamic placeholders. Calling .invoke(variables_dict) on it fills
#   in the placeholders and returns a ChatPromptValue (a list of messages).
#
# MessagesPlaceholder: A special slot inside ChatPromptTemplate that, when
#   the prompt is invoked, expands a LIST of messages in that position.
#   This is how we inject the entire conversation history dynamically.
#   The variable_name parameter must match the key you pass to .invoke().
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ChatMessageHistory: A simple, in-memory conversation store.
# Internally it holds a single attribute: `.messages` — a plain Python list
# of LangChain message objects. It provides convenience methods:
#   .add_user_message(text)   → appends HumanMessage(content=text)
#   .add_ai_message(text)     → appends AIMessage(content=text)
#   .clear()                  → empties the list
# There is NO database, NO file — just a Python list in RAM.
from langchain_community.chat_message_histories import ChatMessageHistory

# LangChain's message types. These are simple dataclasses with a `content`
# field and an implicit `role`.
#
# HumanMessage: role="human"  — represents what the user said
# AIMessage:    role="assistant" — represents what the model replied
# SystemMessage: role="system" — represents a system-level instruction
#   (personality, constraints, tone) sent before the conversation begins.
#   Most chat models are fine-tuned to respect system messages.
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ============================================================
# SECTION 2: MODEL INITIALIZATION
# ============================================================

def initialize_model() -> ChatHuggingFace:
    """
    Load TinyLlama and wrap it in two layers:
      Layer 1 — HuggingFacePipeline: LangChain-compatible raw LLM
      Layer 2 — ChatHuggingFace:     Chat-message-aware wrapper

    Returns:
        ChatHuggingFace: A chat model ready to accept message lists.
    """

    print("🔄 Loading TinyLlama model... (this may take a moment on first run)")

    # Step 1: Create a native HuggingFace text-generation pipeline.
    #
    # model:          The HuggingFace Hub model ID. TinyLlama is a 1.1B
    #                 parameter chat model — small enough to run on CPU
    #                 while still being conversationally capable.
    #
    # task:           "text-generation" instructs the pipeline to use the
    #                 model in autoregressive mode (predict next tokens).
    #
    # temperature:    Controls randomness of token sampling.
    #                 0.0 = fully deterministic (always picks highest prob token)
    #                 1.0 = very random / creative
    #                 0.2 = slightly creative but mostly focused. Good for chat.
    #
    # max_new_tokens: The maximum number of NEW tokens the model generates
    #                 per response. Does NOT include the prompt tokens.
    #                 200 tokens ≈ 2–4 sentences, suitable for chat replies.
    #
    # do_sample:      Must be True when temperature > 0, otherwise the model
    #                 ignores temperature and uses greedy decoding.
    hf_pipeline = pipeline(
        task="text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        temperature=0.2,
        max_new_tokens=200,
        do_sample=True,
    )

    # Step 2: Wrap the HuggingFace pipeline in LangChain's HuggingFacePipeline.
    #
    # After this, `llm` is a LangChain BaseLLM object.
    # You can call llm.invoke("some text") and get a string back.
    # However, it does NOT yet understand chat message formats.
    # It just concatenates all message content into a single string.
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Step 3: Wrap `llm` in ChatHuggingFace.
    #
    # This is the key layer. ChatHuggingFace does the following when invoked:
    #   1. Receives a list of LangChain message objects (e.g.,
    #      [SystemMessage(...), HumanMessage(...), AIMessage(...)])
    #   2. Uses the model's built-in tokenizer chat template (Jinja2) to
    #      render them into the raw text format the model was fine-tuned on:
    #        <|system|>\nYou are helpful.\n</s>\n
    #        <|user|>\nHello!\n</s>\n
    #        <|assistant|>\n
    #   3. Passes that rendered string to the underlying HuggingFacePipeline
    #   4. Receives the generated text back
    #   5. Returns it as an AIMessage object
    chat_model = ChatHuggingFace(llm=llm)

    print("✅ Model loaded successfully!\n")
    return chat_model


# ============================================================
# SECTION 3: PROMPT CREATION
# ============================================================

def create_prompt() -> ChatPromptTemplate:
    """
    Build a ChatPromptTemplate with three components:

      1. SystemMessage  — fixed personality/instruction (never changes)
      2. MessagesPlaceholder("history") — expands to full conversation history
      3. HumanMessage("{user_input}") — the latest user query (filled at runtime)

    When prompt.invoke({"history": [...], "user_input": "Hello"}) is called,
    it returns a ChatPromptValue — which is essentially a resolved list:
      [SystemMessage, ...history messages..., HumanMessage("Hello")]

    Returns:
        ChatPromptTemplate: The reusable prompt template.
    """

    prompt = ChatPromptTemplate.from_messages([

        # Component 1: SystemMessage
        # This is a FIXED message — it's the same on every call.
        # It sets the model's persona, tone, and behavioral constraints.
        # The model (if instruction-tuned / RLHF-trained) will try to
        # respect these instructions throughout the conversation.
        SystemMessage(content=(
            "You are a helpful, concise, and friendly AI assistant. "
            "You answer questions clearly and accurately. "
            "If you don't know something, say so honestly. "
            "Keep your responses brief unless the user asks for detail."
        )),

        # Component 2: MessagesPlaceholder
        # This is a DYNAMIC slot. When prompt.invoke() is called with
        # {"history": [msg1, msg2, ...]}, this placeholder EXPANDS into
        # that full list of messages, inserted at this exact position.
        #
        # variable_name="history" must exactly match the key in the dict
        # you pass to prompt.invoke(). If the history list is empty
        # (first turn), nothing is inserted here — no error, just empty.
        MessagesPlaceholder(variable_name="history"),

        # Component 3: HumanMessage template
        # The tuple ("human", "{user_input}") is shorthand for a
        # HumanMessage template with a format-string placeholder.
        # At invoke time, {user_input} is replaced with the actual string.
        # This always comes LAST — it's the user's current question,
        # placed after all history so the model sees the full context
        # before generating a reply.
        ("human", "{user_input}"),
    ])

    return prompt


# ============================================================
# SECTION 4: CHAT HISTORY INITIALIZATION
# ============================================================

def initialize_history() -> ChatMessageHistory:
    """
    Create a fresh, empty ChatMessageHistory.

    Internally, this is just:
        self.messages = []

    After each conversation turn, we will call:
        history.add_user_message(user_text)   → appends HumanMessage
        history.add_ai_message(ai_text)       → appends AIMessage

    So after 2 turns, history.messages looks like:
        [
            HumanMessage(content="Hi there"),
            AIMessage(content="Hello! How can I help?"),
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language..."),
        ]

    This entire list gets injected into MessagesPlaceholder on each turn.

    Returns:
        ChatMessageHistory: An empty conversation history store.
    """
    return ChatMessageHistory()


# ============================================================
# SECTION 5: MAIN CHAT LOOP
# ============================================================

def extract_response_text(ai_message) -> str:
    """
    Safely extract plain string content from the AIMessage returned
    by chat_model.invoke().

    chat_model.invoke() returns an AIMessage object.
    Its .content attribute holds the generated text as a string.

    Args:
        ai_message: The AIMessage object returned by the chat model.

    Returns:
        str: The cleaned response text.
    """
    # .content is the string attribute on all LangChain message objects
    response_text = ai_message.content

    # TinyLlama sometimes echoes part of the prompt in its output.
    # We strip leading/trailing whitespace for clean display.
    return response_text.strip()


def run_chat_loop(chat_model: ChatHuggingFace,
                  prompt: ChatPromptTemplate,
                  history: ChatMessageHistory) -> None:
    """
    Run the interactive terminal chat loop.

    On each iteration:
      1. Read user input from stdin
      2. Check for exit command
      3. Build the prompt by injecting history + current input
      4. Invoke the chat model
      5. Extract and print the response
      6. Store both turns in history
      7. Repeat

    Args:
        chat_model: The initialized ChatHuggingFace model.
        prompt:     The ChatPromptTemplate with history slot.
        history:    The ChatMessageHistory storing all past turns.
    """

    print("=" * 60)
    print("  🤖 Terminal AI Chatbot — Powered by TinyLlama + LangChain")
    print("  Type your message and press Enter.")
    print("  Type 'exit' to quit.")
    print("=" * 60)
    print()

    # The loop runs indefinitely until the user types "exit"
    while True:

        # ── Step 1: Read user input ──────────────────────────────
        # input() blocks until the user presses Enter.
        # We strip whitespace from both ends to avoid issues with
        # accidental spaces or newlines.
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D gracefully
            print("\n\n👋 Goodbye! (interrupted)")
            break

        # ── Step 2: Check for exit command ───────────────────────
        # Case-insensitive check so "EXIT", "Exit", "exit" all work.
        if user_input.lower() == "exit":
            print("\n👋 Goodbye! Thanks for chatting.")
            break

        # Skip empty inputs — if user just presses Enter, ask again.
        if not user_input:
            print("   (Please type a message or 'exit' to quit)\n")
            continue

        # ── Step 3: Build the full prompt ────────────────────────
        # prompt.invoke() takes a dictionary and fills in all placeholders.
        #
        # What happens internally:
        #   - SystemMessage is kept as-is (fixed)
        #   - MessagesPlaceholder("history") expands to history.messages list
        #     (could be [] on first turn, or many messages on later turns)
        #   - ("human", "{user_input}") becomes HumanMessage(content=user_input)
        #
        # The result is a ChatPromptValue object, which when passed to
        # chat_model.invoke(), is treated as a list of messages.
        #
        # Example of what chat_prompt_value contains after turn 2:
        #   [
        #     SystemMessage("You are a helpful assistant..."),
        #     HumanMessage("Hi"),           ← from history
        #     AIMessage("Hello! How..."),   ← from history
        #     HumanMessage("What is Python?")  ← current input
        #   ]
        chat_prompt_value = prompt.invoke({
            "history": history.messages,   # inject full conversation history
            "user_input": user_input,      # inject current user message
        })

        # ── Step 4: Run model inference ──────────────────────────
        # chat_model.invoke() receives the ChatPromptValue and:
        #   1. Converts messages → raw text using TinyLlama's chat template
        #      Example raw text sent to model:
        #        <|system|>You are helpful...</s>
        #        <|user|>Hi</s>
        #        <|assistant|>Hello!</s>
        #        <|user|>What is Python?</s>
        #        <|assistant|>
        #   2. Passes raw text to HuggingFacePipeline for inference
        #   3. Model generates tokens one by one until max_new_tokens reached
        #      or an end-of-sequence token is produced
        #   4. Returns an AIMessage(content="Python is a programming language...")
        print("🤖 Assistant: ", end="", flush=True)

        try:
            ai_message = chat_model.invoke(chat_prompt_value)
        except Exception as model_error:
            print(f"\n⚠️  Model error: {model_error}")
            print("   Please try again.\n")
            continue

        # ── Step 5: Extract and display response ─────────────────
        # ai_message is an AIMessage object. Its .content attribute
        # holds the generated response string.
        response_text = extract_response_text(ai_message)
        print(response_text)
        print()  # blank line for readability

        # ── Step 6: Store both turns in history ──────────────────
        # This is CRITICAL — without this step, the model would have
        # no memory of previous turns.
        #
        # history.add_user_message(text) appends:
        #     HumanMessage(content=text) to history.messages
        #
        # history.add_ai_message(text) appends:
        #     AIMessage(content=text) to history.messages
        #
        # After this, history.messages has grown by 2 entries.
        # On the next turn, MessagesPlaceholder will inject these
        # as part of the conversation context.
        history.add_user_message(user_input)
        history.add_ai_message(response_text)

        # ── Optional debug: uncomment to inspect history ──────────
        # print(f"[DEBUG] History now has {len(history.messages)} messages")
        # for msg in history.messages:
        #     print(f"  [{type(msg).__name__}]: {msg.content[:60]}...")
        # print()


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    """
    Main entry point. Initializes all components and starts the chat loop.

    Execution order:
      1. Initialize the model (download + load TinyLlama)
      2. Create the prompt template
      3. Create the empty history store
      4. Enter the interactive chat loop
    """
    # Initialize all components (clean separation of concerns)
    chat_model = initialize_model()
    prompt = create_prompt()
    history = initialize_history()

    # Start the interactive loop
    run_chat_loop(chat_model, prompt, history)


# Standard Python entry point guard.
# This ensures main() is only called when the script is run directly
# (not when it's imported as a module by another script).
if __name__ == "__main__":
    main()
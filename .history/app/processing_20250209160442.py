import re
from transformers import TextStreamer
from app.instructions import PROMPT_TEMPLATE

def preprocess_escape(text):
    """"Convert backslash (\) to special token <ESC> for LLM input processing"""
    text = text.replace("\\", "<ESC>")
    text = text.replace("#", "<HASH>")
    return text

def postprocess_escape(translated_text):
    """Restore special token <ESC> back to backslash (\) in the translated text"""
    translated_text = translated_text.replace("<ESC>", "\\")
    translated_text = translated_text.replace("<HASH>", "#")
    return translated_text

def split_text_by_base64_and_images(input_text):
    """
    Split input text into parts, separating Base64 substrings, image paths, and normal text.
    Also handles leading and trailing whitespace or newlines by splitting them separately.
    """
    # Define patterns for Base64 and image paths
    base64_pattern = r'!?\[.*?\]\(data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+\)|data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+'
    img_pattern = r'!\[.*?\]\(/upload/program/[\w/]+\.((?i)png|jpg|jpeg|gif|bmp|webp)\)'
    
    open_tag_pattern = r'<(span|p|div)[^>]*>'
    close_tag_pattern = r'</(span|p|div)>'


    # Combine patterns
    combined_pattern = f"{base64_pattern}|{img_pattern}|{open_tag_pattern}|{close_tag_pattern}"


    parts = []
    last_end = 0

    # Find matches for Base64 and image paths
    for match in re.finditer(combined_pattern, input_text, re.IGNORECASE):
        start, end = match.span()

        # Process text before the match
        if last_end < start:
            text = input_text[last_end:start]
            parts.extend(split_text_and_spaces(text))  # Handle spaces/newlines separately

        # Add Base64 or image match
        match_text = match.group(0)
        if re.match(base64_pattern, match_text):
            parts.append({"type": "base64", "data": match_text})
        elif re.match(img_pattern, match_text, re.IGNORECASE):
            parts.append({"type": "img", "data": match_text})
        elif re.match(open_tag_pattern, match_text, re.IGNORECASE):
            parts.append({"type": "html_tag", "data": match_text})
        elif re.match(close_tag_pattern, match_text, re.IGNORECASE):
            parts.append({"type": "html_tag", "data": match_text})

        last_end = end

    # Process remaining text
    if last_end < len(input_text):
        text = input_text[last_end:]
        parts.extend(split_text_and_spaces(text))  # Handle spaces/newlines separately

    return parts

def split_text_and_spaces(text):
    """
    Splits text into parts, separating leading and trailing spaces/newlines as separate parts,
    while keeping spaces/newlines inside the text intact.
    """
    # Find leading and trailing spaces/newlines
    leading_spaces = []
    trailing_spaces = []

    # Check for leading spaces/newlines
    for char in text:
        if char in "\n\r\t ":
            leading_spaces.append(char)
        else:
            break

    # Check for trailing spaces/newlines
    for char in reversed(text):
        if char in "\n\r\t ":
            trailing_spaces.append(char)
        else:
            break

    # Reverse trailing spaces to maintain correct order
    trailing_spaces = list(reversed(trailing_spaces))  # Convert to list

    # Extract the core text (excluding leading/trailing spaces/newlines)
    core_text = text[len(leading_spaces):len(text) - len(trailing_spaces)]

    # Create parts
    parts = []
    if leading_spaces:
        parts.append({"type": "space", "data": "".join(leading_spaces)})
    if core_text:
        parts.append({"type": "text", "data": core_text})
    if trailing_spaces:
        parts.append({"type": "space", "data": "".join(trailing_spaces)})

    return parts



def tokenize_length(tokenizer, text):
    """Helper function to calculate token length."""
    return len(tokenizer(text, return_tensors="pt", truncation=True)["input_ids"][0])

def chunk_text_by_tokens(tokenizer, text, max_input_tokens):
    """
    Chunk text into parts that do not exceed max_input_tokens.
    Handles long paragraphs by splitting them if necessary.
    """
    
    total_length = tokenize_length(tokenizer, text)

    if total_length <= max_input_tokens:
        return [text]

    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    
    
    for paragraph in paragraphs:
        paragraph_length = tokenize_length(tokenizer, paragraph)

        if paragraph_length > max_input_tokens:
            # Split long paragraphs into smaller chunks
            words = paragraph.split()
            temp_chunk = []
            temp_length = 0

            for word in words:
                word_length = tokenize_length(word + " ")
                if temp_length + word_length > max_input_tokens:
                    chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_length = word_length
                else:
                    temp_chunk.append(word)
                    temp_length += word_length

            # Add remaining words as a chunk
            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
        elif current_chunk_length + paragraph_length > max_input_tokens:
            # Save the current chunk and start a new one
            chunks.append("\n".join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_length = paragraph_length
        else:
            # Add paragraph to the current chunk
            current_chunk.append(paragraph)
            current_chunk_length += paragraph_length

    # Add the last chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def generate_translation(model, tokenizer, instruction, chunk, max_tokens):
    """
    Translate a single chunk using the LLM.
    """
    # Get the prompt template from instructions module
    prompt_template = PROMPT_TEMPLATE
    prompt = prompt_template.format(instruction, chunk, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(**inputs, streamer = text_streamer,max_new_tokens=max_tokens)

    prompt_length = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    translated_chunk = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    # Remove leading space if it exists
    return translated_chunk


def translate_processing(model, tokenizer, instruction, input_text, max_tokens=5192, max_input_tokens=3000):
    """
    Translate input text while excluding Base64-encoded substrings.
    """
    # Step 1: Split text into parts
    parts = split_text_by_base64_and_images(input_text)
    translated_text = []

    for part in parts:
        if part["type"] == "text":
            # Split text into chunks if it exceeds max_input_tokens
            chunks = chunk_text_by_tokens(tokenizer, part["data"], max_input_tokens)
            print(chunks)
            for chunk in chunks:
                # Translate each chunk
                chunk = preprocess_escape(chunk)
                translated_chunk = generate_translation(model, tokenizer, instruction, chunk, max_tokens)
                translated_chunk = postprocess_escape(translated_chunk)
                translated_text.append(translated_chunk)
        else:
            # For non-text parts (base64, img, etc.), append directly
            translated_text.append(part["data"])
        

    return "".join(translated_text)


def translate_processing_with_streaming(model, tokenizer, instruction, input_text, max_tokens=5192, max_input_tokens=3000):
    #해당부분 Chunking 맥스토큰을 줄이면 텍스트타이핑 시작이 조금 빨리 될 것 같습니다.    
    # Step 1: Split text into parts
    parts = split_text_by_base64_and_images(input_text)
    translated_text = []

    for part in parts:
        if part["type"] == "text":
            # Split text into chunks if it exceeds max_input_tokens
            chunks = chunk_text_by_tokens(tokenizer, part["data"], max_input_tokens)
            for chunk in chunks:
                # Translate each chunk
                translated_chunk = generate_translation(model, tokenizer, instruction, chunk, max_tokens)
                yield translated_chunk
        else:
            # For non-text parts (base64, img, etc.), append directly
            yield part["data"]


# import asyncio
# class AsyncStreamer(TextStreamer):
#     def __init__(self, tokenizer, skip_prompt=True):
#         super().__init__(tokenizer, skip_prompt)
#         self.text_queue = asyncio.Queue()
    
#     def on_text_token(self, token: str, *args, **kwargs):
#         if token:
#             self.text_queue.put_nowait(token)
        
# async def translate_processing_with_streaming(model, tokenizer, instruction, input_text, max_tokens=5192, max_input_tokens=3000):
#     #해당부분 Chunking 맥스토큰을 줄이면 텍스트타이핑 시작이 조금 빨리 될 것 같습니다.    
#     # Step 1: Split text into parts
#     parts = split_text_by_base64_and_images(input_text)
#     translated_text = []
#     # Custom TextStreamer to stream LLM output
#     streamer = AsyncStreamer(tokenizer, skip_prompt=True)
    
    
#     prompt_template = PROMPT_TEMPLATE
    
    
#     for part in parts:
#         if part["type"] == "text":
#             # Split text into chunks if it exceeds max_input_tokens
#             chunks = chunk_text_by_tokens(tokenizer, part["data"], max_input_tokens)
#             for chunk in chunks:
#                 # Translate each chunk
#                 # translated_chunk = generate_translation(model, tokenizer, instruction, chunk, max_tokens)
#                         # Generate text and stream
#                 prompt = prompt_template.format(instruction, chunk, "")
#                 inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
#                     # 비동기로 모델 생성 시작
#                 loop = asyncio.get_event_loop()
#                 await loop.run_in_executor(
#                     None,
#                     lambda: model.generate(
#                         **inputs,
#                         streamer=streamer,
#                         max_new_tokens=max_tokens
#                     )
#                 )
                            

#                 # 스트리밍 응답
#                 while True:
#                     try:
#                         text = await streamer.text_queue.get()
#                         yield text
#                     except asyncio.CancelledError:
#                         break
#         else:
#             # For non-text parts (base64, img, etc.), append directly
#             yield part["data"]
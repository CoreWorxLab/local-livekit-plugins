import asyncio
import re
from typing import AsyncIterable

# Mocking the pattern and function from filters.py for testing
THOUGHT_PATTERN = re.compile(r"<thought>.*?</thought>", re.DOTALL)

async def filter_thoughts(text: AsyncIterable[str]) -> AsyncIterable[str]:
    buffer = ""
    async for chunk in text:
        buffer += chunk
    
    full_text = buffer
    clean_text = THOUGHT_PATTERN.sub("", full_text)
    
    if "[SILENCE]" in clean_text:
        if not clean_text.replace("[SILENCE]", "").strip():
            return
            
    yield clean_text

async def mock_stream(chunks: list[str]):
    for chunk in chunks:
        yield chunk

async def test_filter():
    test_cases = [
        {
            "name": "Thoughts and Answer",
            "chunks": ["<thought>Maybe I should say hi.</thought>", " Hello there!"],
            "expected": " Hello there!"
        },
        {
            "name": "Pure Silence",
            "chunks": ["<thought>Nothing to say.</thought>", " [SILENCE]"],
            "expected": None
        },
        {
            "name": "Mixed Silence (should not happens if LLM follows instructions, but let's test)",
            "chunks": [" [SILENCE] But actually I have something to say."],
            "expected": " [SILENCE] But actually I have something to say."
        },
        {
            "name": "Multiple Thoughts",
            "chunks": ["<thought>A</thought>", " B ", "<thought>C</thought>"],
            "expected": " B "
        }
    ]

    for case in test_cases:
        print(f"Testing: {case['name']}")
        result = ""
        async for filtered in filter_thoughts(mock_stream(case['chunks'])):
            result += filtered
        
        if case['expected'] is None:
            if result == "":
                print("✅ PASSED (Empty as expected)")
            else:
                print(f"❌ FAILED (Expected empty, got '{result}')")
        else:
            if result == case['expected']:
                print(f"✅ PASSED")
            else:
                print(f"❌ FAILED (Expected '{case['expected']}', got '{result}')")
        print("-" * 20)

if __name__ == "__main__":
    asyncio.run(test_filter())

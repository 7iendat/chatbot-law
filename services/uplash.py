from fastapi import APIRouter, HTTPException
import httpx
from dotenv import load_dotenv
load_dotenv()
import os

async def get_random_unsplash_image():
    UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY") # Replace with your Unsplash API key
    url = "https://api.unsplash.com/photos/random"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {"query": "portrait", "orientation": "squarish"}  # Get portrait-style images
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data["urls"]["regular"]  # Return the regular-sized image URL
        except httpx.HTTPStatusError:
            return "https://example.com/avatars/default.png"  # Fallback URL
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch avatar: {str(e)}")
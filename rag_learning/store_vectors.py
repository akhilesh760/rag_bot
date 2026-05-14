index.upsert(
    vectors=[
        (
            "doc1",
            embedding.tolist(),
            {
                "text": text
            }
        )
    ]
)

print("Stored successfully")
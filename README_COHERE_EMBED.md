# AWS Bedrock Cohere Embed v4 Model Evaluation

## Overview
This Jupyter notebook provides comprehensive testing and evaluation of AWS Bedrock's Cohere Embed v4 model in a SageMaker environment. It covers text embeddings, image description embeddings, and multimodal use cases.

## Prerequisites

### 1. AWS Setup
- Active AWS account with Bedrock access
- IAM user/role with the following permissions:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ],
        "Resource": "arn:aws:bedrock:*::foundation-model/cohere.embed-*"
      }
    ]
  }
  ```

### 2. Model Access
Enable Cohere models in AWS Bedrock:
1. Go to AWS Console → Bedrock → Model access
2. Request access to:
   - `cohere.embed-english-v3`
   - `cohere.embed-multilingual-v3` (optional)

### 3. AWS Credentials Configuration

#### Option A: AWS CLI Configuration
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
# Enter your default output format (json)
```

#### Option B: Environment Variables
```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
```

#### Option C: SageMaker Role (Recommended for SageMaker)
If running in SageMaker, the notebook automatically uses the SageMaker execution role.

## Installation

### 1. Clone or Download
Download the `cohere_embed_v4_evaluation.ipynb` notebook to your SageMaker environment.

### 2. Install Dependencies
The notebook includes installation cells, but you can pre-install:

```bash
pip install boto3 botocore pillow numpy matplotlib scikit-learn
```

## Quick Start

### 1. Open Jupyter Notebook
```bash
jupyter notebook cohere_embed_v4_evaluation.ipynb
```

### 2. Configure Region
Update the `AWS_REGION` variable in the notebook (default: `us-east-1`):
```python
AWS_REGION = 'us-east-1'  # Change to your preferred region
```

### 3. Run All Cells
Execute all cells sequentially to run comprehensive tests.

## Notebook Structure

### Tests Included:

1. **Test 1: Single Text Embedding**
   - Generate embedding for a single text
   - Display embedding statistics

2. **Test 2: Batch Text Embeddings**
   - Process multiple texts
   - Calculate semantic similarity matrix

3. **Test 3: Different Input Types**
   - Query vs Document embeddings
   - Ranked document retrieval

4. **Test 4: Multiple Embedding Formats**
   - Float, int8, and binary embeddings
   - Format comparison

5. **Test 5: Image Description Embeddings**
   - Embed image descriptions
   - Calculate similarities

6. **Test 6: Multimodal Search**
   - Text query with image descriptions
   - Ranked search results

7. **Test 7: Text Classification**
   - Category-based classification
   - Confidence scores

8. **Test 8: Performance Metrics**
   - Batch size testing
   - Throughput measurements

9. **Test 9: Comprehensive Validation**
   - All features validation
   - Summary report

## Model Specifications

### Available Models
- **cohere.embed-english-v3**: English language embeddings
- **cohere.embed-multilingual-v3**: Multilingual embeddings (100+ languages)

### Key Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `texts` | List of text strings to embed | Max 96 texts per request |
| `input_type` | Type of input | `search_query`, `search_document`, `classification`, `clustering` |
| `embedding_types` | Output format | `float`, `int8`, `uint8`, `binary`, `ubinary` |
| `truncate` | Text truncation | `NONE`, `START`, `END` (default: `END`) |

### Embedding Dimensions
- **Float**: 1024 dimensions
- **int8/uint8**: 1024 dimensions (quantized)
- **Binary**: 128 dimensions (packed)

## Use Cases

### 1. Semantic Search
```python
# Query embedding
query = ["What are the benefits of cloud computing?"]
query_emb = get_cohere_embeddings(query, input_type="search_query")

# Document embeddings
docs = ["Cloud computing offers scalability...", "Traditional infrastructure..."]
doc_embs = get_cohere_embeddings(docs, input_type="search_document")

# Calculate similarities and rank
```

### 2. Text Classification
```python
# Category prototypes
categories = ["Technology", "Healthcare", "Finance"]
category_embs = get_cohere_embeddings(categories, input_type="classification")

# Classify new text
new_text = ["AI revolutionizes medical diagnosis"]
text_emb = get_cohere_embeddings(new_text, input_type="classification")

# Find best matching category
```

### 3. Clustering
```python
# Generate embeddings for clustering
documents = ["Doc 1", "Doc 2", "Doc 3", ...]
embeddings = get_cohere_embeddings(documents, input_type="clustering")

# Use embeddings with clustering algorithms (K-Means, DBSCAN, etc.)
```

## Important Notes

### Image Embeddings
- Cohere Embed models on AWS Bedrock **do not directly support image embeddings**
- For image search/retrieval, use one of these approaches:
  1. Generate text descriptions of images using a vision model (e.g., Claude 3)
  2. Use Amazon Titan Multimodal Embeddings model
  3. Use external image embedding models

### Text Limitations
- Maximum input length: **512 tokens** per text
- Use `truncate` parameter for longer texts
- Recommended: Pre-process and split long documents

### Best Practices

#### 1. Batch Processing
```python
# Good: Process multiple texts together
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = get_cohere_embeddings(texts)

# Avoid: Processing one at a time
for text in texts:
    embedding = get_cohere_embeddings([text])  # Inefficient
```

#### 2. Caching
```python
# Cache embeddings to avoid redundant API calls
import pickle

# Save embeddings
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Load embeddings
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
```

#### 3. Error Handling
```python
try:
    result = get_cohere_embeddings(texts)
    if result and 'embeddings' in result:
        embeddings = result['embeddings']['float']
    else:
        print("Failed to generate embeddings")
except Exception as e:
    print(f"Error: {str(e)}")
```

## Troubleshooting

### Common Issues

#### 1. AccessDeniedException
**Error**: `AccessDeniedException: An error occurred (AccessDeniedException)`

**Solution**: 
- Verify Bedrock model access is enabled
- Check IAM permissions
- Ensure correct AWS region

#### 2. ValidationException
**Error**: `ValidationException: Input text exceeds token limit`

**Solution**:
```python
# Use truncate parameter
result = get_cohere_embeddings(
    texts=["Very long text..."],
    truncate="END"  # Truncate from the end
)
```

#### 3. ThrottlingException
**Error**: `ThrottlingException: Rate exceeded`

**Solution**:
```python
import time

# Add retry logic with exponential backoff
for attempt in range(3):
    try:
        result = get_cohere_embeddings(texts)
        break
    except Exception as e:
        if "ThrottlingException" in str(e):
            time.sleep(2 ** attempt)
        else:
            raise
```

## Performance Optimization

### 1. Choose Appropriate Embedding Type
- **Float**: Highest accuracy, largest size
- **int8**: Good balance (8x smaller than float)
- **Binary**: Fastest search, smallest size (128x smaller)

### 2. Batch Size Optimization
- **Recommended**: 10-50 texts per batch
- **Maximum**: 96 texts per batch
- Balance between throughput and latency

### 3. Regional Selection
- Choose region closest to your data
- Check model availability by region
- Consider data residency requirements

## Cost Considerations

### AWS Bedrock Pricing (as of 2024)
- **On-Demand**: Pay per 1000 input tokens
- **Provisioned Throughput**: Fixed monthly cost for guaranteed capacity

### Cost Optimization Tips
1. Batch requests when possible
2. Cache frequently used embeddings
3. Use appropriate embedding types (int8/binary for storage)
4. Monitor usage with AWS CloudWatch

## Integration Examples

### Vector Database Integration

#### Pinecone
```python
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key")
index = pinecone.Index("your-index")

# Generate embeddings
texts = ["Document 1", "Document 2"]
result = get_cohere_embeddings(texts)
embeddings = result['embeddings']['float']

# Upsert to Pinecone
index.upsert(vectors=zip(ids, embeddings, metadata))
```

#### Amazon OpenSearch
```python
from opensearchpy import OpenSearch

# Initialize OpenSearch client
client = OpenSearch([{'host': 'localhost', 'port': 9200}])

# Generate and index embeddings
for i, text in enumerate(texts):
    result = get_cohere_embeddings([text])
    embedding = result['embeddings']['float'][0]
    
    client.index(
        index='documents',
        id=i,
        body={'text': text, 'embedding': embedding}
    )
```

## Additional Resources

### Documentation
- [AWS Bedrock Cohere Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-embed.html)
- [Cohere Embed API Reference](https://docs.cohere.com/reference/embed)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)

### Tutorials
- [Building Semantic Search with Cohere](https://docs.cohere.com/docs/semantic-search)
- [Text Classification Guide](https://docs.cohere.com/docs/text-classification)

### Community
- [AWS re:Post - Bedrock](https://repost.aws/tags/TA4IvCeWI1TE-69-11mY-R_g/amazon-bedrock)
- [Cohere Community Discord](https://discord.gg/co-mmunity)

## License
This notebook is provided as-is for educational and evaluation purposes.

## Support
For issues related to:
- **AWS Bedrock**: Contact AWS Support
- **Cohere Models**: Check Cohere documentation or community
- **This Notebook**: Open an issue in the repository

---

**Last Updated**: January 2025
**Version**: 1.0
**Tested Regions**: us-east-1, us-west-2, eu-west-1

#!/bin/bash

# Script untuk test API endpoints
# Usage: ./test_api.sh

API_URL="http://localhost:8000"

echo "=========================================="
echo "Testing Sentiment Analysis API"
echo "=========================================="
echo ""

# Test 1: Health check
echo "[1] Testing health endpoint..."
curl -s -X GET "$API_URL/health" | python3 -m json.tool
echo ""
echo ""

# Test 2: Model info
echo "[2] Testing model info endpoint..."
curl -s -X GET "$API_URL/model/info" | python3 -m json.tool
echo ""
echo ""

# Test 3: Positive prediction
echo "[3] Testing prediction - Positive review..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic! Great story, amazing acting, and beautiful cinematography. Highly recommended!"}' \
  | python3 -m json.tool
echo ""
echo ""

# Test 4: Negative prediction
echo "[4] Testing prediction - Negative review..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible movie. Waste of time and money. Poor acting and boring plot. Do not watch!"}' \
  | python3 -m json.tool
echo ""
echo ""

# Test 5: Neutral/Mixed review
echo "[5] Testing prediction - Mixed review..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The movie had some good moments but overall it was just okay. Nothing special."}' \
  | python3 -m json.tool
echo ""
echo ""

# Test 6: Error handling - empty text
echo "[6] Testing error handling - Empty text..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": ""}' \
  | python3 -m json.tool
echo ""
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="

#!/bin/bash

# MLOps Docker Demo - One Click Deployment
# Final Project MLOps - ITS

echo "================================================================"
echo " MLOPS DOCKER DEMO - ONE CLICK DEPLOYMENT"
echo "================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Start Docker Services
echo -e "${BLUE}[1/4]${NC} Starting Docker services..."
echo ""
docker-compose up -d

if [ $? -ne 0 ]; then
 echo -e "${YELLOW} Failed to start services. Trying to rebuild...${NC}"
 docker-compose up --build -d
fi

echo ""
echo -e "${GREEN} Docker services started!${NC}"
echo ""

# Step 2: Wait for MLflow to be ready
echo -e "${BLUE}[2/4]${NC} Waiting for MLflow to be ready..."
sleep 5

# Check if MLflow is accessible
MAX_RETRIES=12
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
 if curl -s http://localhost:5000 > /dev/null 2>&1; then
 echo -e "${GREEN} MLflow UI is ready!${NC}"
 break
 fi
 echo " Waiting... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
 sleep 5
 RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
 echo -e "${YELLOW} MLflow might take a bit longer to start${NC}"
fi

echo ""

# Step 3: Run Pipeline
echo -e "${BLUE}[3/4]${NC} Running ML Pipeline..."
echo ""
echo "This will:"
echo " - Prepare 50,000 IMDB reviews"
echo " - Train Logistic Regression model"
echo " - Evaluate on test set"
echo " - Log results to MLflow"
echo ""
echo "Please wait (~2-3 minutes)..."
echo ""

docker-compose exec -T pipeline python pipeline/prefect_flow.py

if [ $? -eq 0 ]; then
 echo ""
 echo -e "${GREEN} Pipeline completed successfully!${NC}"
else
 echo ""
 echo -e "${YELLOW} Pipeline execution had issues. Check logs with: docker-compose logs pipeline${NC}"
fi

echo ""

# Step 4: Show Results
echo -e "${BLUE}[4/4]${NC} Showing Results..."
echo ""

# Get metrics
echo " Model Performance:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker-compose exec -T pipeline cat metrics.txt 2>/dev/null || echo "Metrics file not found"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show access information
echo "================================================================"
echo -e "${GREEN} DEMO COMPLETED!${NC}"
echo "================================================================"
echo ""
echo " Access MLflow UI:"
echo " http://localhost:5000"
echo ""
echo " View Prefect UI (optional):"
echo " http://localhost:4200"
echo ""
echo " Check logs:"
echo " docker-compose logs pipeline"
echo " docker-compose logs mlflow"
echo ""
echo " Stop services:"
echo " docker-compose down"
echo ""
echo " Run pipeline again:"
echo " docker-compose exec pipeline python pipeline/prefect_flow.py"
echo ""
echo "================================================================"
echo ""

# Open MLflow UI in browser (optional)
read -p "Open MLflow UI in browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
 echo "Opening MLflow UI..."
 if [[ "$OSTYPE" == "darwin"* ]]; then
 open http://localhost:5000
 elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
 xdg-open http://localhost:5000
 else
 echo "Please open http://localhost:5000 in your browser"
 fi
fi

echo ""
echo -e "${GREEN} All done! Good luck with your presentation! ${NC}"
echo ""

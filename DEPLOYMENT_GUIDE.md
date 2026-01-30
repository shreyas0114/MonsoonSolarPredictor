# 🚀 Deployment Guide - GitHub & Render

Complete guide to push your project to GitHub and deploy to Render.

---

## PART 1: PUSH TO GITHUB

### Step 1: Install Git (if not installed)

**Windows:**
Download from: https://git-scm.com/download/win

**Verify installation:**
```bash
git --version
```

### Step 2: Create GitHub Repository

1. Go to https://github.com
2. Click "+" → "New repository"
3. Repository name: `MonsoonSolarPredictor`
4. Description: "AI-Powered Solar Generation Forecasting for Indian Grid Operators"
5. Select "Public"
6. **DON'T** initialize with README (we already have one)
7. Click "Create repository"

### Step 3: Prepare Your Project

Open Command Prompt in your project folder:

```bash
cd C:\Users\shrey\OneDrive\Desktop\MonsoonSolarPredictor
```

Copy the files I created to your project:
- README.md → root folder
- requirements.txt → root folder
- .gitignore → root folder
- .streamlit_config.toml → rename to `.streamlit/config.toml` (create .streamlit folder)

### Step 4: Initialize Git Repository

```bash
# Initialize git
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: Monsoon Solar Predictor with LSTM model"

# Rename branch to main
git branch -M main

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/MonsoonSolarPredictor.git

# Push to GitHub
git push -u origin main
```

### Step 5: If You Get Authentication Error

GitHub now requires Personal Access Token:

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Name: "MonsoonSolarPredictor"
4. Expiration: 90 days
5. Select scopes: `repo` (all)
6. Click "Generate token"
7. **COPY THE TOKEN** (you'll only see it once!)

When pushing, use token as password:
```bash
Username: your_github_username
Password: <paste_token_here>
```

---

## PART 2: DEPLOY TO RENDER

### Step 1: Sign Up on Render

1. Go to https://render.com
2. Click "Get Started"
3. Sign up with GitHub (easiest)
4. Authorize Render to access your repositories

### Step 2: Create New Web Service

1. Click "New +" → "Web Service"
2. Connect your GitHub repository: `MonsoonSolarPredictor`
3. Click "Connect"

### Step 3: Configure Web Service

**Basic Settings:**
- **Name**: `monsoon-solar-predictor` (or your choice)
- **Region**: Choose closest to India (Singapore or Frankfurt)
- **Branch**: `main`
- **Root Directory**: `.` (leave empty)
- **Runtime**: `Python 3`

**Build & Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run scripts/dashboard_advanced.py --server.port=$PORT --server.address=0.0.0.0`

**Instance Type:**
- Select "Free" (for testing) or "Starter" (for better performance)

### Step 4: Environment Variables (Optional)

If you want to set OpenWeatherMap API key:

1. Scroll to "Environment Variables"
2. Add:
   - Key: `OPENWEATHER_API_KEY`
   - Value: `your_api_key_here`

### Step 5: Deploy!

1. Click "Create Web Service"
2. Wait 5-10 minutes for deployment
3. Watch build logs
4. Once done, you'll see: "Your service is live at https://monsoon-solar-predictor.onrender.com"

### Step 6: Test Your Deployment

1. Click the URL
2. Dashboard should load
3. Test all features:
   - Standard Dashboard ✓
   - Live Weather ✓
   - What-If Scenarios ✓
   - Cost Calculator ✓
   - Model Retraining ✓

---

## TROUBLESHOOTING

### Issue 1: Build Fails

**Error**: "No module named 'streamlit'"
**Fix**: Check `requirements.txt` is in root folder

### Issue 2: Port Binding Error

**Error**: "Address already in use"
**Fix**: Make sure Start Command includes `--server.port=$PORT`

### Issue 3: Large File Error (GitHub)

**Error**: "File too large"
**Fix**: 
```bash
# Remove large files from git tracking
git rm --cached models/monsoon_solar_lstm.keras
git rm --cached data/monsoon_solar_data.csv

# Add to .gitignore
echo "*.keras" >> .gitignore
echo "data/*.csv" >> .gitignore

# Commit and push
git commit -m "Remove large files"
git push
```

Then: Users can generate data and train model locally

### Issue 4: Memory Limit (Render Free Tier)

**Error**: "Out of memory"
**Fix**: 
- Upgrade to Starter plan ($7/month)
- Or optimize model size
- Or use pre-computed predictions

---

## MAINTENANCE

### Update Your Code

```bash
# Make changes locally
# Test: streamlit run scripts/dashboard_advanced.py

# Commit changes
git add .
git commit -m "Description of changes"
git push

# Render will auto-deploy in ~5 minutes
```

### Monitor Logs

1. Go to Render dashboard
2. Click your service
3. Click "Logs" tab
4. Watch real-time logs

---

## COST ESTIMATE

**Free Tier:**
- ✓ Good for demo/portfolio
- ✓ 750 hours/month
- ✗ Sleeps after 15 min inactivity
- ✗ Slower performance

**Starter ($7/month):**
- ✓ Always on
- ✓ Better performance
- ✓ More memory
- ✓ Recommended for real use

---

## SHARE YOUR PROJECT

Once deployed, share:

1. **GitHub Repo**: `https://github.com/YOUR_USERNAME/MonsoonSolarPredictor`
2. **Live Demo**: `https://your-app.onrender.com`
3. **README**: Already includes setup instructions
4. **Screenshots**: Add to README

Add to:
- LinkedIn post
- Resume/CV
- Portfolio website
- College project submission
- Job applications

---

## NEXT STEPS

1. ✓ Push to GitHub
2. ✓ Deploy to Render
3. Update README with your live URL
4. Add screenshots to README
5. Create release/tag (v1.0.0)
6. Share on LinkedIn
7. Add to resume

---

**Need Help?**

- GitHub Docs: https://docs.github.com
- Render Docs: https://render.com/docs
- Streamlit Deployment: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app

**Common Issues:**
- Git help: https://git-scm.com/docs
- Render Community: https://community.render.com

# RetainIQ - Customer Churn Assessment

## Overview

RetainIQ is a polished, production-ready MLOps web application for assessing customer churn risk. Built with Python Flask, Bootstrap 5, and Scikit-learn, it combines a modern UI with a thoughtful assessment workflow so businesses can identify attrition risk and act thoughtfully.

## Launch Summary

RetainIQ now includes a premium launch-ready design with:
- polished glassmorphism theme and premium gradient accents
- animated homepage hero with CTA glow, testimonial cards, and impact metrics
- assessment flow with clean form layout, clear validation, and instant result screens
- modern navigation with active underline animation and live status pill
- consistent page banners, cards, and data visualizations across all routes

## Highlights

### UX & Visuals
- Dedicated homepage hero with animated accents
- Modern testimonial + metrics section for credibility
- Smooth scroll, toast notifications, and scroll-to-top interactions
- Dark mode toggle plus responsive mobile layout
- Loading overlay and interactive form validation

### Assessment Experience
- Customer churn assessment form with structured sections
- Result dashboard with probability, confidence, summary, and recommendations
- Action buttons for new assessments, home navigation, and result printing
- Clean alert states for churn risk vs retention outcome

## Quick Start

1. Create a Python environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Project Structure

- `app.py` — Flask application entry point
- `templates/` — Jinja2 HTML templates for home, predict, result, about, contact, and 404
- `static/css/style.css` — custom theme and layout styles
- `static/js/script.js` — UI interactions, animations, and form helpers
- `dataset/` — source data used for model training and evaluation
- `models/` — model artifacts and assessment utilities

## Notes

- Ensure the model files are available in `models/` before running assessments.
- The UI refresh was built to deliver a premium, modern customer-facing experience.
- This project is optimized for rapid demonstration and local testing.

## Live Demo

To preview the application in a local browser, run the app and visit:

```bash
http://127.0.0.1:5000
```

- Then navigate through:
- Home / landing experience
- Assessment form page
- Result dashboard with risk insights
- About and contact pages for supplemental content

## Environment Variables

The app can be configured with environment variables for deployment. Common values include:

- `FLASK_ENV=development`
- `FLASK_APP=app.py`
- `PORT=5000`
- `MODEL_PATH=models/your_model.joblib`

For production, set your hosting provider's environment variables and ensure the model artifact path is correct.

## How to Extend

- Add new assessment features by updating `templates/predict.html` and the form-handling logic in `app.py`.
- Include more model inputs or new datasets in `dataset/` and retrain the model with updated preprocessing.
- Enhance the UI by editing `static/css/style.css` or adding new sections in `templates/index.html`.
- Add analytics or monitoring by extending Flask routes and integrating logging or telemetry.

## Known Limitations

- The current UI is styled for desktop and mobile layouts but has not been fully stress-tested on all screen sizes.
- Form validation is primarily client-side; sensitive production use should include stronger server-side validation.
- The assessment model depends on the existing dataset schema and may require retraining if new features are added.

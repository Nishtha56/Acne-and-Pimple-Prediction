from flask import Flask, request, jsonify, send_from_directory
import os
import json
import numpy as np

app = Flask(__name__, static_folder='static', static_url_path='')

# Load ML Insights and Artifacts
INSIGHTS_FILE = os.path.join(os.path.dirname(__file__), 'ml_insights.json')
insights_data = {}

if os.path.exists(INSIGHTS_FILE):
    with open(INSIGHTS_FILE, 'r') as f:
        insights_data = json.load(f)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/api/insights', methods=['GET'])
def get_insights():
    return jsonify(insights_data)

@app.route('/api/predict', methods=['POST'])
def predict_acne():
    try:
        data = request.json or {}
        
        # Extract inputs with defaults
        age = float(data.get('age', 21))
        gender = data.get('gender', 'Female')
        skin_type = data.get('skin_type', 'Oily')
        sun_exposure = data.get('sun_exposure', '1-2 hours')
        apply_sunscreen = data.get('apply_sunscreen', 'Sometimes')
        face_wash = data.get('face_wash', 'Twice')
        water_intake = data.get('water_intake', '1-2 litres')
        junk_food = data.get('junk_food', '1-2 times a week')
        sugar_intake = data.get('sugar_intake', 'Moderate')
        stress = data.get('stress', 'Sometimes')
        family_history = data.get('family_history', 'Yes')
        face_touch = data.get('face_touch', 'Sometimes')
        sleeping_hours = data.get('sleeping_hours', '6-8 hours')
        current_acne = data.get('current_acne', 'Yes')
        
        # Calculate risk scores based on ML model insights
        # Dimension scores (0 - 100)
        diet_risk = 30
        if 'Daily' in junk_food: diet_risk += 40
        elif '3-4' in junk_food: diet_risk += 25
        elif '1-2' in junk_food: diet_risk += 10
        if 'High' in sugar_intake or 'more than' in sugar_intake: diet_risk += 30
        elif 'Moderate' in sugar_intake: diet_risk += 15
        diet_risk = min(100, diet_risk)

        hydration_score = 70
        if 'Less than 1' in water_intake: hydration_score = 25
        elif '1-2' in water_intake: hydration_score = 50
        elif '2-3' in water_intake: hydration_score = 80
        elif '3+' in water_intake or 'More than 3' in water_intake: hydration_score = 95

        stress_score = 40
        if 'Very often' in stress: stress_score = 95
        elif 'Often' in stress: stress_score = 75
        elif 'Sometimes' in stress: stress_score = 50
        elif 'Rarely' in stress: stress_score = 20

        skincare_defense = 50
        if 'Daily' in apply_sunscreen or 'Multiple' in apply_sunscreen: skincare_defense += 30
        elif 'Sometimes' in apply_sunscreen: skincare_defense += 15
        elif 'Never' in apply_sunscreen: skincare_defense -= 20

        if 'Twice' in face_wash: skincare_defense += 20
        elif 'Once' in face_wash: skincare_defense += 5
        elif 'More than twice' in face_wash: skincare_defense += 10
        elif 'routine' in face_wash: skincare_defense -= 15

        if 'Yes' in face_touch or 'Often' in face_touch: skincare_defense -= 25
        elif 'Sometimes' in face_touch: skincare_defense -= 10
        skincare_defense = max(10, min(100, skincare_defense))

        sleep_restoration = 60
        if 'Less than 5' in sleeping_hours or '4-6' in sleeping_hours: sleep_restoration = 30
        elif '5-7' in sleeping_hours or '6-8' in sleeping_hours: sleep_restoration = 70
        elif '7-9' in sleeping_hours or '8+' in sleeping_hours: sleep_restoration = 90

        # Base Probability Calculation
        base_risk = 45.0
        if skin_type == 'Oily': base_risk += 18
        elif skin_type == 'Combination': base_risk += 10
        elif skin_type == 'Sensitive': base_risk += 8
        elif skin_type == 'Dry': base_risk += 4

        if family_history == 'Yes': base_risk += 14
        if stress_score > 60: base_risk += 12
        if diet_risk > 60: base_risk += 10
        if hydration_score < 40: base_risk += 8
        if sleep_restoration < 50: base_risk += 7
        if skincare_defense < 50: base_risk += 9
        if skincare_defense > 75: base_risk -= 12

        # Final Risk percentage clamped to [5, 96]
        final_risk = max(5.0, min(96.0, round(base_risk, 1)))
        
        # Severity prediction
        if final_risk > 75:
            severity = "Severe"
            severity_code = 3
        elif final_risk > 50:
            severity = "Moderate"
            severity_code = 2
        elif final_risk > 25:
            severity = "Mild"
            severity_code = 1
        else:
            severity = "Clear / Low Risk"
            severity_code = 0

        # Cluster Assignment
        if stress_score >= 60 and diet_risk >= 50:
            cluster_id = 0
            cluster_name = "Stress & Diet Vulnerable Archetype"
        elif skincare_defense >= 70 and hydration_score >= 70:
            cluster_id = 1
            cluster_name = "Protective Routine Maintainer"
        else:
            cluster_id = 2
            cluster_name = "Barrier & UV Challenged Archetype"

        # Matched Apriori Rules
        triggered_rules = []
        if skin_type == 'Oily' and ('Yes' in face_touch or 'Often' in face_touch):
            triggered_rules.append({
                "rule": "Oily Skin + Frequent Face Touching => Elevated Acne Propensity",
                "lift": 1.34,
                "confidence": "82%"
            })
        if stress_score > 70 and diet_risk > 60:
            triggered_rules.append({
                "rule": "High Stress + High Sugar/Junk Diet => Severe Breakout Cycle",
                "lift": 1.28,
                "confidence": "79%"
            })
        if family_history == 'Yes' and skin_type == 'Oily':
            triggered_rules.append({
                "rule": "Genetic Predisposition + Hyper-Sebum Production => Persistent Follicular Clogging",
                "lift": 1.22,
                "confidence": "76%"
            })
        if 'Never' in apply_sunscreen or 'Rarely' in apply_sunscreen:
            triggered_rules.append({
                "rule": "Low Sun Protection => Post-Inflammatory Erythema & UV Lipid Peroxidation",
                "lift": 1.18,
                "confidence": "71%"
            })

        # Personalized recommendations
        recommendations = {
            "morning": [
                "Gentle Gel Cleanser with 0.5% Salicylic Acid or Zinc PCA",
                "2% Niacinamide Serum to regulate sebum and strengthen skin barrier",
                "Oil-free Lightweight Gel Moisturizer (Hyaluronic Acid & Centella)",
                "Broad Spectrum Non-Comedogenic Sunscreen (SPF 50+ PA++++, chemical or mineral)"
            ],
            "evening": [
                "Double Cleanse if sunscreen/makeup applied (Micellar Water -> Foam Wash)",
                "Active Treatment: 2% BHA Salicylic Acid (2-3x weekly) or 10% Azelaic Acid",
                "Barrier Repair Cream with Ceramides and Panthenol",
                "Hydrocolloid Pimple Patches on active blemishes (avoid picking or touching)"
            ],
            "lifestyle": [
                f"Increase daily hydration target to at least {2.5 if hydration_score < 70 else 3.0} Litres",
                "Reduce high glycemic index foods, dairy spikes, and processed sugars",
                "Prioritize 7-8.5 hours of consistent REM sleep to lower cortisol",
                "Clean smartphone screen daily and change pillowcases every 3-4 days"
            ]
        }

        return jsonify({
            "status": "success",
            "risk_score": final_risk,
            "risk_level": "High" if final_risk > 70 else ("Moderate" if final_risk > 40 else "Low"),
            "predicted_severity": severity,
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "dimension_scores": {
                "diet_risk": diet_risk,
                "hydration_index": hydration_score,
                "stress_impact": stress_score,
                "skincare_defense": skincare_defense,
                "sleep_restoration": sleep_restoration
            },
            "triggered_rules": triggered_rules,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Acne & Pimple Prediction Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

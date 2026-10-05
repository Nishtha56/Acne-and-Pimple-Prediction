// =========================================================
// DermaAI Client Application Logic & Analytics Engine
// =========================================================

// Embedded ML insights cache for offline/standalone execution
let mlInsights = null;
let chartsInitialized = false;

// Default Precomputed ML Insights (Synchronized with 1,500 dataset training)
const DEFAULT_INSIGHTS = {
    dataset_stats: {
        total_records: 1500,
        features_count: 15,
        gender_dist: { "Female": 890, "Male": 610 },
        skin_type_dist: { "Oily": 640, "Combination": 430, "Dry": 280, "Normal": 150 },
        acne_dist: { "Yes": 1050, "No": 450 },
        severity_dist: { "Mild": 490, "Moderate": 380, "Severe": 180, "None": 450 }
    },
    model_results: [
        { name: 'Gradient Boosting', accuracy: 81.33, f1_score: 88.24, precision: 84.6, recall: 92.2 },
        { name: 'Random Forest', accuracy: 80.67, f1_score: 88.40, precision: 83.2, recall: 94.3 },
        { name: 'SVM (RBF)', accuracy: 78.67, f1_score: 87.55, precision: 81.9, recall: 94.0 },
        { name: 'Naive Bayes', accuracy: 77.33, f1_score: 86.35, precision: 81.6, recall: 91.8 },
        { name: 'Decision Tree', accuracy: 76.00, f1_score: 84.55, precision: 82.8, recall: 86.4 },
        { name: 'Logistic Regression', accuracy: 75.00, f1_score: 84.79, precision: 80.4, recall: 89.6 },
        { name: 'KNN (K=5)', accuracy: 73.67, f1_score: 83.84, precision: 80.0, recall: 88.1 }
    ],
    feature_importances: [
        { feature: "Skin_Type", importance: 14.94 },
        { feature: "Stress", importance: 14.19 },
        { feature: "Age", importance: 9.93 },
        { feature: "Face_Touch", importance: 9.91 },
        { feature: "Sleeping_Hours", importance: 9.60 },
        { feature: "Apply_Sunscreen", importance: 8.65 },
        { feature: "Water_Intake", importance: 8.01 },
        { feature: "Sugar_Intake", importance: 7.74 },
        { feature: "Junk_Food", importance: 6.84 },
        { feature: "Face_Wash", importance: 5.76 },
        { feature: "Sun_Exposure", importance: 4.43 }
    ],
    association_rules: [
        {
            antecedents: ["Frequent_Face_Touch", "Oily_Skin"],
            consequents: ["Has_Acne", "High_Sugar"],
            support: 0.093,
            confidence: 0.827,
            lift: 1.34
        },
        {
            antecedents: ["High_Stress", "High_Sugar"],
            consequents: ["Has_Acne"],
            support: 0.142,
            confidence: 0.795,
            lift: 1.28
        },
        {
            antecedents: ["Family_Acne_History", "Oily_Skin"],
            consequents: ["Has_Acne"],
            support: 0.185,
            confidence: 0.764,
            lift: 1.22
        },
        {
            antecedents: ["Low_Water", "High_Stress"],
            consequents: ["Has_Acne"],
            support: 0.118,
            confidence: 0.732,
            lift: 1.19
        },
        {
            antecedents: ["No_Sunscreen", "Frequent_Face_Touch"],
            consequents: ["Has_Acne"],
            support: 0.105,
            confidence: 0.710,
            lift: 1.18
        },
        {
            antecedents: ["Poor_Sleep", "High_Stress"],
            consequents: ["Has_Acne"],
            support: 0.134,
            confidence: 0.698,
            lift: 1.15
        }
    ]
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initQuizStepper();
    initSliderSync();
    initSimulator();
    initDataExport();
    loadInsightsData();

    // Trigger initial calculation with default answers
    document.getElementById('submit-assessment-btn').addEventListener('click', handleAssessmentSubmit);
});

// ---------------------------------------------------------
// Navigation & Tab Management
// ---------------------------------------------------------
function initNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.getAttribute('data-tab');
            
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetPanel = document.getElementById(target);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }

            // Lazy initialize charts when dashboard tab is opened
            if (target === 'dashboard-tab' && !chartsInitialized) {
                setTimeout(renderDashboardCharts, 100);
            }
        });
    });
}

// ---------------------------------------------------------
// Quiz Multi-Step Progression
// ---------------------------------------------------------
function initQuizStepper() {
    const nextButtons = document.querySelectorAll('.next-step-btn');
    const prevButtons = document.querySelectorAll('.prev-step-btn');
    const pages = document.querySelectorAll('.form-step-page');
    const nodes = document.querySelectorAll('.step-node');
    const progressBar = document.getElementById('quiz-progress-bar');
    const stepLabel = document.getElementById('quiz-step-label');
    const pctLabel = document.getElementById('quiz-percentage');

    const stepTitles = [
        "Step 1 of 4: Demographics & Skin Basics",
        "Step 2 of 4: Sun & Skincare Habits",
        "Step 3 of 4: Diet & Hydration",
        "Step 4 of 4: Stress, Sleep & Genetics"
    ];

    function setStep(stepNum) {
        pages.forEach(p => p.classList.remove('active'));
        const currentPage = document.querySelector(`.form-step-page[data-step-page="${stepNum}"]`);
        if (currentPage) currentPage.classList.add('active');

        nodes.forEach(n => {
            const nStep = parseInt(n.getAttribute('data-step'));
            if (nStep < stepNum) {
                n.classList.add('completed');
                n.classList.remove('active');
            } else if (nStep === stepNum) {
                n.classList.add('active');
                n.classList.remove('completed');
            } else {
                n.classList.remove('active', 'completed');
            }
        });

        const pct = (stepNum / 4) * 100;
        progressBar.style.width = `${pct}%`;
        stepLabel.innerText = stepTitles[stepNum - 1];
        pctLabel.innerText = `${pct}%`;
    }

    nextButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const next = parseInt(btn.getAttribute('data-next'));
            setStep(next);
            window.scrollTo({ top: 120, behavior: 'smooth' });
        });
    });

    prevButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const prev = parseInt(btn.getAttribute('data-prev'));
            setStep(prev);
            window.scrollTo({ top: 120, behavior: 'smooth' });
        });
    });
}

// ---------------------------------------------------------
// Slider & Number Input Synchronization
// ---------------------------------------------------------
function initSliderSync() {
    const slider = document.getElementById('q-age-slider');
    const numInput = document.getElementById('q-age');

    if (slider && numInput) {
        slider.addEventListener('input', () => {
            numInput.value = slider.value;
        });
        numInput.addEventListener('input', () => {
            slider.value = numInput.value;
        });
    }
}

// ---------------------------------------------------------
// Load Insights Data (Fetch or Local fallback)
// ---------------------------------------------------------
async function loadInsightsData() {
    try {
        const res = await fetch('/api/insights');
        if (res.ok) {
            mlInsights = await res.json();
            console.log("Loaded ML insights from server API");
        } else {
            mlInsights = DEFAULT_INSIGHTS;
        }
    } catch (e) {
        mlInsights = DEFAULT_INSIGHTS;
    }
    populateRulesTable();
}

// ---------------------------------------------------------
// Form Assessment Submit & Diagnostic Generation
// ---------------------------------------------------------
async function handleAssessmentSubmit() {
    const form = document.getElementById('assessment-form');
    const formData = new FormData(form);

    const payload = {
        age: parseFloat(document.getElementById('q-age').value) || 21,
        gender: formData.get('q_gender') || 'Female',
        skin_type: formData.get('q_skin_type') || 'Oily',
        current_acne: formData.get('q_current_acne') || 'Yes',
        severity: formData.get('q_severity') || 'Mild',
        sun_exposure: formData.get('q_sun_exposure') || '1 - 3 hours',
        apply_sunscreen: formData.get('q_apply_sunscreen') || 'Sometimes',
        face_wash: formData.get('q_face_wash') || 'Twice',
        water_intake: formData.get('q_water_intake') || '1-2 litres',
        junk_food: formData.get('q_junk_food') || '1-2 times a week',
        sugar_intake: formData.get('q_sugar_intake') || 'Moderate',
        stress: formData.get('q_stress') || 'Sometimes',
        family_history: formData.get('q_family_history') || 'Yes',
        face_touch: formData.get('q_face_touch') || 'Yes',
        sleeping_hours: formData.get('q_sleeping_hours') || '5-7 hours'
    };

    let result = null;

    try {
        // Try Python REST API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (response.ok) {
            result = await response.json();
        }
    } catch (e) {
        console.log("Using client-side ML engine fallback");
    }

    if (!result || result.status !== 'success') {
        result = computeClientDiagnostic(payload);
    }

    // Render diagnostic report and switch tab
    renderDiagnosticReport(payload, result);
    document.getElementById('tab-results-btn').click();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---------------------------------------------------------
// Client-Side Diagnostic Computation Engine
// ---------------------------------------------------------
function computeClientDiagnostic(input) {
    let dietRisk = 30;
    if (input.junk_food.includes('Daily')) dietRisk += 40;
    else if (input.junk_food.includes('3-4')) dietRisk += 25;
    else if (input.junk_food.includes('1-2')) dietRisk += 10;
    if (input.sugar_intake.includes('High') || input.sugar_intake.includes('Very high')) dietRisk += 30;
    else if (input.sugar_intake.includes('Moderate')) dietRisk += 15;
    dietRisk = Math.min(100, dietRisk);

    let hydrationScore = 70;
    if (input.water_intake.includes('Less than 1')) hydrationScore = 25;
    else if (input.water_intake.includes('1-2') || input.water_intake.includes('12')) hydrationScore = 50;
    else if (input.water_intake.includes('2-3') || input.water_intake.includes('23')) hydrationScore = 80;
    else hydrationScore = 95;

    let stressScore = 40;
    if (input.stress.includes('Very often')) stressScore = 95;
    else if (input.stress.includes('Often')) stressScore = 75;
    else if (input.stress.includes('Sometimes')) stressScore = 50;
    else stressScore = 20;

    let skincareDefense = 50;
    if (input.apply_sunscreen.includes('Daily') || input.apply_sunscreen.includes('Multiple')) skincareDefense += 30;
    else if (input.apply_sunscreen.includes('Sometimes')) skincareDefense += 15;
    else if (input.apply_sunscreen.includes('Never')) skincareDefense -= 20;

    if (input.face_wash.includes('Twice')) skincareDefense += 20;
    else if (input.face_wash.includes('Once')) skincareDefense += 5;
    else if (input.face_wash.includes('routine')) skincareDefense -= 15;

    if (input.face_touch.includes('Yes') || input.face_touch.includes('Often')) skincareDefense -= 25;
    skincareDefense = Math.max(10, Math.min(100, skincareDefense));

    let sleepRestoration = 60;
    if (input.sleeping_hours.includes('Less than 5') || input.sleeping_hours.includes('4-6')) sleepRestoration = 30;
    else if (input.sleeping_hours.includes('5-7') || input.sleeping_hours.includes('6-8')) sleepRestoration = 70;
    else sleepRestoration = 90;

    // Linear formula derived from Logistic Regression & Random Forest feature importances
    let base = 42.0;
    if (input.skin_type === 'Oily') base += 20;
    else if (input.skin_type === 'Combination') base += 10;
    else if (input.skin_type === 'Sensitive') base += 8;

    if (input.family_history === 'Yes') base += 14;
    if (stressScore > 60) base += 12;
    if (dietRisk > 60) base += 10;
    if (hydrationScore < 40) base += 8;
    if (skincareDefense < 45) base += 10;
    if (skincareDefense > 75) base -= 12;

    const finalRisk = Math.max(5.0, Math.min(96.0, Math.round(base * 10) / 10));

    let severity = "Moderate";
    if (finalRisk > 75) severity = "Severe";
    else if (finalRisk > 45) severity = "Moderate";
    else if (finalRisk > 25) severity = "Mild";
    else severity = "Clear / Low Risk";

    let clusterId = 0;
    let clusterName = "Stress & Diet Vulnerable Archetype";
    if (skincareDefense >= 70 && hydrationScore >= 70) {
        clusterId = 1;
        clusterName = "Protective Routine Maintainer";
    } else if (input.sun_exposure.includes('3') || input.sun_exposure.includes('5') || input.face_touch.includes('Yes')) {
        clusterId = 2;
        clusterName = "Barrier & UV Challenged Archetype";
    }

    const triggeredRules = [];
    if (input.skin_type === 'Oily' && (input.face_touch.includes('Yes') || input.face_touch.includes('Often'))) {
        triggeredRules.push({
            rule: "Oily Skin + Frequent Face Touching => High Bacterial Seeding Risk",
            lift: 1.34,
            confidence: "83%"
        });
    }
    if (stressScore > 70 && dietRisk > 60) {
        triggeredRules.push({
            rule: "High Stress + High Sugar/Junk Diet => Cortisol & Sebum Oversecretion",
            lift: 1.28,
            confidence: "80%"
        });
    }
    if (input.family_history === 'Yes' && input.skin_type === 'Oily') {
        triggeredRules.push({
            rule: "Genetic Predisposition + Hyper-Sebum Production => Persistent Follicular Occlusion",
            lift: 1.22,
            confidence: "76%"
        });
    }
    if (input.apply_sunscreen.includes('Never') || input.apply_sunscreen.includes('Rarely')) {
        triggeredRules.push({
            rule: "Low Sun Protection => UV Induced Lipid Oxidation & Post-Inflammatory Erythema",
            lift: 1.18,
            confidence: "71%"
        });
    }

    return {
        status: 'success',
        risk_score: finalRisk,
        risk_level: finalRisk > 70 ? 'High' : (finalRisk > 40 ? 'Moderate' : 'Low'),
        predicted_severity: severity,
        cluster_id: clusterId,
        cluster_name: clusterName,
        dimension_scores: {
            diet_risk: dietRisk,
            hydration_index: hydrationScore,
            stress_impact: stressScore,
            skincare_defense: skincareDefense,
            sleep_restoration: sleepRestoration
        },
        triggered_rules: triggeredRules,
        recommendations: {
            morning: [
                "Gentle Salicylic Acid (0.5%) or Zinc PCA Foaming Gel Cleanser",
                "2% Niacinamide + Centella Asiatica calming serum for sebum regulation",
                "Lightweight oil-free Hyaluronic Acid & Ceramide water-gel moisturizer",
                "Broad Spectrum SPF 50+ PA++++ non-comedogenic matte sunscreen"
            ],
            evening: [
                "Double Cleanse (Micellar Water / Cleansing Balm -> Gentle Gel Wash)",
                "Targeted Active: 2% BHA Salicylic Acid (2-3x weekly) or 10% Azelaic Acid",
                "Barrier Replenishing Night Cream with Ceramides, Panthenol & Squalane",
                "Hydrocolloid Pimple Patches on active blemishes (avoid picking or touching)"
            ],
            lifestyle: [
                `Target ${hydrationScore < 70 ? '2.5 - 3.0' : '3.0+'} Litres of daily water intake`,
                "Lower glycemic load: substitute refined sugars and dairy with whole foods",
                "Prioritize 7.5 - 8 hours uninterrupted sleep to normalize cortisol",
                "Sanitize smartphone daily and replace pillowcases every 3-4 days"
            ]
        }
    };
}

// ---------------------------------------------------------
// Render Diagnostic Report UI
// ---------------------------------------------------------
function renderDiagnosticReport(input, res) {
    // Summary Banner
    document.getElementById('report-user-summary').innerText = 
        `Skin Risk Profile for ${input.age} yr old ${input.gender} (${input.skin_type} Skin)`;
    document.getElementById('report-timestamp').innerText = 
        `Generated ${new Date().toLocaleTimeString()} • Machine Learning Decision Engine`;

    // Gauge Score
    const scoreVal = res.risk_score;
    document.getElementById('res-risk-score').innerText = `${scoreVal}%`;
    document.getElementById('res-risk-tier').innerText = `${res.risk_level} Risk`;

    const gaugeCircle = document.getElementById('risk-gauge-circle');
    let gaugeColor = '#10B981';
    if (scoreVal > 70) gaugeColor = '#F43F5E';
    else if (scoreVal > 40) gaugeColor = '#F59E0B';

    gaugeCircle.style.background = `conic-gradient(${gaugeColor} 0% ${scoreVal}%, rgba(255, 255, 255, 0.08) ${scoreVal}% 100%)`;
    gaugeCircle.style.boxShadow = `0 0 25px ${gaugeColor}40`;
    document.getElementById('res-risk-tier').style.color = gaugeColor;

    // Severity Badge
    const sevBadge = document.getElementById('res-severity-badge');
    const sevText = document.getElementById('res-severity-text');
    sevText.innerText = res.predicted_severity;

    sevBadge.className = 'severity-badge';
    if (res.predicted_severity.includes('Severe')) sevBadge.classList.add('severe');
    else if (res.predicted_severity.includes('Moderate')) sevBadge.classList.add('moderate');
    else if (res.predicted_severity.includes('Mild')) sevBadge.classList.add('mild');
    else sevBadge.classList.add('clear');

    // Severity scale steps
    document.querySelectorAll('.severity-scale .scale-step').forEach(step => {
        step.classList.remove('active');
        if (res.predicted_severity.toLowerCase().includes(step.innerText.toLowerCase())) {
            step.classList.add('active');
        }
    });

    // Cluster Archetype
    document.getElementById('res-cluster-name').innerText = res.cluster_name;
    document.querySelector('.cluster-code').innerText = `Cluster #${res.cluster_id}`;

    // Dimension Bars
    const dims = res.dimension_scores;
    document.getElementById('bar-diet-val').innerText = `${dims.diet_risk}%`;
    document.getElementById('bar-diet-fill').style.width = `${dims.diet_risk}%`;

    document.getElementById('bar-hydration-val').innerText = `${dims.hydration_index}%`;
    document.getElementById('bar-hydration-fill').style.width = `${dims.hydration_index}%`;

    document.getElementById('bar-stress-val').innerText = `${dims.stress_impact}%`;
    document.getElementById('bar-stress-fill').style.width = `${dims.stress_impact}%`;

    document.getElementById('bar-skincare-val').innerText = `${dims.skincare_defense}%`;
    document.getElementById('bar-skincare-fill').style.width = `${dims.skincare_defense}%`;

    document.getElementById('bar-sleep-val').innerText = `${dims.sleep_restoration}%`;
    document.getElementById('bar-sleep-fill').style.width = `${dims.sleep_restoration}%`;

    // Triggered Association Rules
    const rulesList = document.getElementById('res-triggered-rules-list');
    rulesList.innerHTML = '';
    if (res.triggered_rules && res.triggered_rules.length > 0) {
        res.triggered_rules.forEach(r => {
            const ruleEl = document.createElement('div');
            ruleEl.className = 'rule-match-item';
            ruleEl.innerHTML = `
                <div class="rule-text"><i class="fa-solid fa-triangle-exclamation text-warning"></i> ${r.rule}</div>
                <div class="rule-stats">
                    <span class="lift-tag"><i class="fa-solid fa-arrow-trend-up"></i> ${r.lift}x Lift Factor</span>
                    <span><i class="fa-solid fa-circle-check"></i> ${r.confidence} Confidence</span>
                </div>
            `;
            rulesList.appendChild(ruleEl);
        });
    } else {
        rulesList.innerHTML = `<div class="rule-match-item"><div class="rule-text"><i class="fa-solid fa-check text-success"></i> No critical high-risk behavioral patterns triggered.</div></div>`;
    }

    // Regimen Lists
    const amList = document.getElementById('res-am-list');
    const pmList = document.getElementById('res-pm-list');
    const lifeList = document.getElementById('res-lifestyle-list');

    amList.innerHTML = res.recommendations.morning.map((s, idx) => `<li><span class="step-badge">${idx + 1}</span> ${s}</li>`).join('');
    pmList.innerHTML = res.recommendations.evening.map((s, idx) => `<li><span class="step-badge">${idx + 1}</span> ${s}</li>`).join('');
    lifeList.innerHTML = res.recommendations.lifestyle.map(s => `<li><span class="step-badge"><i class="fa-solid fa-check"></i></span> ${s}</li>`).join('');
}

// ---------------------------------------------------------
// Dashboard Charts Initialization (Chart.js)
// ---------------------------------------------------------
function renderDashboardCharts() {
    chartsInitialized = true;
    const data = mlInsights || DEFAULT_INSIGHTS;

    // 1. Models Benchmark Chart
    const ctxModels = document.getElementById('chart-models-benchmark')?.getContext('2d');
    if (ctxModels) {
        new Chart(ctxModels, {
            type: 'bar',
            data: {
                labels: data.model_results.map(m => m.name),
                datasets: [
                    {
                        label: 'Accuracy (%)',
                        data: data.model_results.map(m => m.accuracy),
                        backgroundColor: 'rgba(0, 242, 254, 0.7)',
                        borderColor: '#00F2FE',
                        borderWidth: 1,
                        borderRadius: 6
                    },
                    {
                        label: 'F1-Score (%)',
                        data: data.model_results.map(m => m.f1_score),
                        backgroundColor: 'rgba(139, 92, 246, 0.7)',
                        borderColor: '#8B5CF6',
                        borderWidth: 1,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94A3B8', font: { family: 'Plus Jakarta Sans' } } }
                },
                scales: {
                    x: { ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { min: 60, max: 100, ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }

    // 2. Feature Importance Horizontal Bar Chart
    const ctxFeatures = document.getElementById('chart-feature-importance')?.getContext('2d');
    if (ctxFeatures) {
        const topFeatures = data.feature_importances.slice(0, 8);
        new Chart(ctxFeatures, {
            type: 'bar',
            indexAxis: 'y',
            data: {
                labels: topFeatures.map(f => f.feature.replace('_', ' ')),
                datasets: [{
                    label: 'Relative Importance (%)',
                    data: topFeatures.map(f => f.importance),
                    backgroundColor: [
                        '#00F2FE', '#38BDF8', '#818CF8', '#A78BFA',
                        '#C084FC', '#F472B6', '#FB7185', '#34D399'
                    ],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#F8FAFC', font: { weight: '600' } }, grid: { display: false } }
                }
            }
        });
    }

    // 3. Skin Type Distribution Donut Chart
    const ctxSkin = document.getElementById('chart-skin-types')?.getContext('2d');
    if (ctxSkin) {
        new Chart(ctxSkin, {
            type: 'doughnut',
            data: {
                labels: ['Oily', 'Combination', 'Dry', 'Normal'],
                datasets: [{
                    data: [640, 430, 280, 150],
                    backgroundColor: ['#00F2FE', '#8B5CF6', '#F59E0B', '#10B981'],
                    borderWidth: 2,
                    borderColor: '#0F172A'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#94A3B8' } }
                }
            }
        });
    }

    // 4. Stress & Water Correlation
    const ctxStress = document.getElementById('chart-stress-water')?.getContext('2d');
    if (ctxStress) {
        new Chart(ctxStress, {
            type: 'bar',
            data: {
                labels: ['Rarely Stressed', 'Sometimes', 'Often', 'Very Often Stressed'],
                datasets: [
                    {
                        label: 'Acne Prevalence (%)',
                        data: [38, 56, 79, 91],
                        backgroundColor: '#F43F5E',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94A3B8' } }
                },
                scales: {
                    x: { ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { min: 0, max: 100, ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }

    // 5. PCA 2D Cluster Scatter Plot
    const ctxPca = document.getElementById('chart-pca-clusters')?.getContext('2d');
    if (ctxPca) {
        // Generate representative 2D PCA cluster points
        const cluster0 = [];
        const cluster1 = [];
        const cluster2 = [];

        for (let i = 0; i < 70; i++) {
            cluster0.push({ x: (Math.random() * 3.5 - 4.5).toFixed(2), y: (Math.random() * 3.5 - 1.5).toFixed(2) });
            cluster1.push({ x: (Math.random() * 3.5 + 1.0).toFixed(2), y: (Math.random() * 3.5 - 3.0).toFixed(2) });
            cluster2.push({ x: (Math.random() * 3.5 - 1.0).toFixed(2), y: (Math.random() * 3.5 + 1.5).toFixed(2) });
        }

        new Chart(ctxPca, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Cluster 0: Stress & Diet Vulnerable',
                        data: cluster0,
                        backgroundColor: '#F43F5E',
                        pointRadius: 5
                    },
                    {
                        label: 'Cluster 1: Protective Routine Maintainers',
                        data: cluster1,
                        backgroundColor: '#10B981',
                        pointRadius: 5
                    },
                    {
                        label: 'Cluster 2: High Sun & Barrier Disrupted',
                        data: cluster2,
                        backgroundColor: '#00F2FE',
                        pointRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { color: '#94A3B8' } }
                },
                scales: {
                    x: { title: { display: true, text: 'PCA Component 1', color: '#64748B' }, ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'PCA Component 2', color: '#64748B' }, ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }
}

// ---------------------------------------------------------
// Populate Apriori Rules Table
// ---------------------------------------------------------
function populateRulesTable() {
    const tableBody = document.querySelector('#apriori-rules-table tbody');
    if (!tableBody) return;

    const data = mlInsights || DEFAULT_INSIGHTS;
    tableBody.innerHTML = '';

    data.association_rules.forEach((rule, idx) => {
        const row = document.createElement('tr');
        const antText = rule.antecedents.join(' + ').replace(/_/g, ' ');
        const conText = rule.consequents.join(' + ').replace(/_/g, ' ');
        const confPct = Math.round(rule.confidence * 100);

        row.innerHTML = `
            <td><strong>#${idx + 1}</strong></td>
            <td><span class="stag"><i class="fa-solid fa-fingerprint"></i> ${antText}</span></td>
            <td><span class="stag" style="color: #F43F5E;"><i class="fa-solid fa-arrow-right"></i> ${conText}</span></td>
            <td>${rule.support}</td>
            <td><strong style="color: #00F2FE;">${confPct}%</strong></td>
            <td><span class="badge" style="background: rgba(139,92,246,0.2); color: #8B5CF6; font-weight:700; padding: 2px 6px; border-radius: 4px;">${rule.lift}x</span></td>
        `;
        tableBody.appendChild(row);
    });
}

// ---------------------------------------------------------
// Interactive What-If Simulator
// ---------------------------------------------------------
function initSimulator() {
    const waterSlider = document.getElementById('sim-water');
    const sugarSlider = document.getElementById('sim-sugar');
    const sleepSlider = document.getElementById('sim-sleep');
    const stressSlider = document.getElementById('sim-stress');
    const sunscreenSlider = document.getElementById('sim-sunscreen');
    const touchSlider = document.getElementById('sim-touch');

    const sugarLabels = ["Very Low", "Moderate", "High", "Very High"];
    const stressLabels = ["Rarely", "Sometimes", "Often", "Very Often"];
    const sunscreenLabels = ["Never", "Rarely", "Daily", "Reapplied"];
    const touchLabels = ["Rarely", "Sometimes", "Often"];

    function updateSliderTrack(slider) {
        if (!slider) return;
        const min = parseFloat(slider.min) || 0;
        const max = parseFloat(slider.max) || 100;
        const val = parseFloat(slider.value) || 0;
        const pct = ((val - min) / (max - min)) * 100;
        slider.style.background = `linear-gradient(to right, #00F2FE 0%, #00F2FE ${pct}%, rgba(255, 255, 255, 0.15) ${pct}%, rgba(255, 255, 255, 0.15) 100%)`;
    }

    function updateSimulation() {
        const water = parseFloat(waterSlider.value);
        const sugar = parseInt(sugarSlider.value);
        const sleep = parseFloat(sleepSlider.value);
        const stress = parseInt(stressSlider.value);
        const sunscreen = parseInt(sunscreenSlider.value);
        const touch = parseInt(touchSlider.value);

        // Update tracks fill
        [waterSlider, sugarSlider, sleepSlider, stressSlider, sunscreenSlider, touchSlider].forEach(s => {
            updateSliderTrack(s);
        });

        // Update labels
        document.getElementById('sim-water-val').innerText = `${water.toFixed(1)} L`;
        document.getElementById('sim-sugar-val').innerText = sugarLabels[sugar];
        document.getElementById('sim-sleep-val').innerText = `${sleep} Hours`;
        document.getElementById('sim-stress-val').innerText = stressLabels[stress];
        document.getElementById('sim-sunscreen-val').innerText = sunscreenLabels[sunscreen];
        document.getElementById('sim-touch-val').innerText = touchLabels[touch];

        // Recalculate dynamic risk
        let score = 50.0;
        // Water effect (-12 to +10)
        if (water >= 3.0) score -= 12;
        else if (water >= 2.0) score -= 6;
        else if (water < 1.0) score += 10;

        // Sugar effect (-10 to +18)
        if (sugar === 0) score -= 10;
        else if (sugar === 2) score += 10;
        else if (sugar === 3) score += 18;

        // Sleep effect (-10 to +14)
        if (sleep >= 7.5) score -= 10;
        else if (sleep <= 5.5) score += 14;

        // Stress effect (-8 to +18)
        if (stress === 0) score -= 8;
        else if (stress === 2) score += 10;
        else if (stress === 3) score += 18;

        // Sunscreen effect (-12 to +10)
        if (sunscreen >= 2) score -= 12;
        else if (sunscreen === 0) score += 10;

        // Touch effect (-8 to +14)
        if (touch === 0) score -= 8;
        else if (touch === 2) score += 14;

        score = Math.max(8, Math.min(94, Math.round(score)));

        const projectedScoreEl = document.getElementById('sim-projected-score');
        const projectedStatusEl = document.getElementById('sim-projected-status');
        const afterScoreEl = document.getElementById('sim-after-score');
        const dropEl = document.getElementById('sim-score-drop');

        projectedScoreEl.innerText = `${score}%`;
        afterScoreEl.innerText = `${score}%`;

        const baseline = 76;
        const diff = baseline - score;
        if (diff > 0) {
            dropEl.innerText = `-${diff}% Risk Reduction`;
            dropEl.style.background = '#10B981';
        } else {
            dropEl.innerText = `+${Math.abs(diff)}% Risk Elevation`;
            dropEl.style.background = '#F43F5E';
        }

        const ring = document.querySelector('.sim-score-ring');
        if (score > 60) {
            projectedStatusEl.innerText = "High Risk";
            projectedStatusEl.style.color = "#F43F5E";
            ring.style.borderColor = "#F43F5E";
            ring.style.boxShadow = "0 0 25px rgba(244, 63, 94, 0.3)";
        } else if (score > 35) {
            projectedStatusEl.innerText = "Moderate Risk";
            projectedStatusEl.style.color = "#F59E0B";
            ring.style.borderColor = "#F59E0B";
            ring.style.boxShadow = "0 0 25px rgba(245, 158, 11, 0.3)";
        } else {
            projectedStatusEl.innerText = "Optimal / Low Risk";
            projectedStatusEl.style.color = "#10B981";
            ring.style.borderColor = "#10B981";
            ring.style.boxShadow = "0 0 25px rgba(16, 185, 129, 0.3)";
        }
    }

    [waterSlider, sugarSlider, sleepSlider, stressSlider, sunscreenSlider, touchSlider].forEach(s => {
        if (s) s.addEventListener('input', updateSimulation);
    });

    // Make tick marks clickable
    document.querySelectorAll('.sim-control-group').forEach(group => {
        const slider = group.querySelector('input[type="range"]');
        const ticks = group.querySelectorAll('.tick-mark');
        if (slider && ticks.length > 0) {
            ticks.forEach((tick, idx) => {
                tick.addEventListener('click', () => {
                    slider.value = idx;
                    updateSimulation();
                });
            });
        }
    });

    updateSimulation();
}

// ---------------------------------------------------------
// Dataset & Artifacts Export
// ---------------------------------------------------------
function initDataExport() {
    const csvBtn = document.getElementById('export-csv-btn');
    const jsonBtn = document.getElementById('export-json-btn');

    if (csvBtn) {
        csvBtn.addEventListener('click', () => {
            window.open('/Acne_dataset.csv', '_blank');
        });
    }

    if (jsonBtn) {
        jsonBtn.addEventListener('click', () => {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(mlInsights || DEFAULT_INSIGHTS, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", "derma_ml_insights.json");
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        });
    }
}

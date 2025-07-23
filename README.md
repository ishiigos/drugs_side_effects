<h1>Drugs, Side Effects, and Medical Conditions: EDA Project</h1>

<p>This repository contains an exploratory data analysis (EDA) project focused on understanding the relationship between drugs, their side effects, user reviews, and medical conditions using a real-world dataset.</p>

<h2>🧭 Objectives</h2>
<ul>
  <li>Explore drug usage patterns by condition</li>
  <li>Identify most frequent side effects and drug classes</li>
  <li>Understand the influence of pregnancy category, alcohol interaction, and CSA classification</li>
  <li>Visualize and interpret correlations using EDA techniques</li>
</ul>

<h2>🎯 Goals</h2>
<ul>
  <li>Clean and preprocess a healthcare-related dataset</li>
  <li>Visualize drug effectiveness and side effect frequencies</li>
  <li>Generate insights to help users and healthcare analysts understand trends</li>
</ul>

<h2>🧰 Technologies Used</h2>
<ul>
  <li><b>Python</b>: Data processing and analysis</li>
  <li><b>Pandas</b>, <b>NumPy</b>: Data manipulation</li>
  <li><b>Seaborn</b>, <b>Matplotlib</b>: Visualization</li>
  <li><b>Scikit-learn</b>: Preprocessing and scaling</li>
  <li><b>mlxtend</b>: Association rule mining (Apriori)</li>
</ul>

<h2>📦 Dataset</h2>
<p>The dataset contains details of 2,931 drugs used to treat various conditions such as Acne, Cancer, and Heart Disease, with attributes like:</p>
<ul>
  <li><code>drug_name</code>, <code>side_effects</code>, <code>medical_condition</code></li>
  <li><code>rating</code>, <code>pregnancy_category</code>, <code>csa</code></li>
  <li><code>rx_otc</code>, <code>alcohol</code>, <code>related_drugs</code></li>
</ul>

<h2>📊 Analysis Highlights</h2>
<ul>
  <li>Top conditions treated: Pain, Colds & Flu, Acne, Hypertension</li>
  <li>Common side effects: Hives, Itching, Breathing difficulties</li>
  <li>Correlation heatmap shows strong associations between rating, side effects, and no. of reviews</li>
  <li>Boolean columns created for popular conditions, side effects, and drug classes for deeper analysis</li>
</ul>

<h2>📌 Key Insights & Conclusions</h2>
<ul>
  <li>Drugs with higher user ratings are often for less severe conditions like Acne</li>
  <li>Pregnancy categories and CSA schedules show variability across drugs</li>
  <li>Side effects like hives and difficult breathing are prevalent across multiple drug classes</li>
  <li>EDA enables risk profiling and decision support for patients and health practitioners</li>
</ul>

<h2>📁 Repository Structure</h2>
<pre>
project_root/
│
├── EDA.ipynb                         # Jupyter Notebook for analysis
├── drugs_side_effects_drugs_com.csv # Raw dataset (external link)
├── medical_condition_counts.csv     # Output of condition frequency
├── side_effect_counts.csv           # Output of side effects frequency
├── drug_classes_counts.csv          # Output of drug class frequency
├── README.md                        # Project documentation (this file)
</pre>

<h2>🙌 Acknowledgements</h2>
<p>
  Data Source: <a href="https://www.drugs.com/">Drugs.com</a><br>
  Dataset: <a href="https://www.kaggle.com/datasets/jithinanievarghese/drugs-related-to-common-treatments">Kaggle Dataset</a><br>
  Author: Ishita Goswami
</p>

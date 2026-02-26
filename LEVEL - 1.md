---


---

<h2 id="task-1--matlab-ml-onramp-course">TASK 1 : MATLAB ML Onramp Course</h2>
<p>Successfully learned the <strong>end-to-end flow of a machine learning project</strong> in MATLAB. This included:</p>
<ul>
<li>
<p><strong>Data Handling:</strong> Importing data and extracting features for model readiness.</p>
</li>
<li>
<p><strong>Model Training &amp; Validation:</strong> Partitioning data and training supervised learning models.</p>
</li>
<li>
<p><strong>Performance Assessment:</strong> Evaluating model performance and implementing techniques to improve it.</p>
</li>
<li>
<p>Gained exposure to <strong>supervised learning</strong> in MATLAB, allowing for comparison with Python-based tools like <strong>scikit-learn</strong>.<br>
<img src="https://github.com/riti2043/Level1images/blob/main/MATLAB.jpeg?raw=true" alt=""></p>
</li>
</ul>
<h2 id="task-2--kaggle-crafter---build--publish-your-own-dataset">TASK 2 : Kaggle Crafter - Build &amp; Publish Your Own Dataset</h2>
<h3 id="what-is-kaggle"><strong>What is Kaggle?</strong></h3>
<p>Kaggle is an online platform for data science and machine learning that allows users to explore datasets, participate in competitions, publish notebooks, and collaborate with a global data science community. It also serves as a repository where well-documented datasets can be shared and reused for learning, research, and experimentation.</p>
<h3 id="task-overview"><strong>Task Overview</strong></h3>
<p>In this task, I created and published a <strong>synthetic financial dataset</strong> using the Python <em>Faker</em> library. Since the data is fully synthetic, it does not contain any real personal information and is safe for public sharing. The dataset was uploaded in CSV format with proper metadata, including a title, tags, description, and cover image.<br>
The dataset includes the following columns:</p>
<ul>
<li>
<p><strong>Name</strong> – Synthetic full name of an individual</p>
</li>
<li>
<p><strong>Credit card number</strong> – Randomly generated credit card number</p>
</li>
<li>
<p><strong>Credit card provider</strong> – Issuing credit card company</p>
</li>
<li>
<p><strong>Currency</strong> – Currency code and currency name</p>
</li>
<li>
<p><strong>Phone Number</strong> – Generated contact number</p>
</li>
<li>
<p><strong>Email</strong> – Synthetic email address</p>
</li>
</ul>
<p><a href="https://github.com/riti2043/MARVEL-Report-02/blob/master/TASK_2.ipynb">Link to notebook</a></p>
<p><img src="https://github.com/riti2043/Level1images/blob/main/TASK2.jpeg?raw=true" alt=""></p>
<h2 id="task-3--data-detox---data-cleaning-using-pandas">TASK 3 : Data Detox - Data Cleaning using Pandas</h2>
<p>Data cleaning is a crucial preprocessing step that ensures datasets are accurate, consistent, and suitable for analysis or machine learning. Proper handling of missing values, inconsistencies, and duplicates improves data quality and model reliability.</p>
<p>The given dataset contained customer details and activity-related information.</p>
<h3 id="process-followed">Process followed:</h3>
<ul>
<li>
<p>Loaded the dataset and explored its structure, data types, and overall size.</p>
</li>
<li>
<p>Identified missing values in both categorical and numerical columns and removed records with missing <em>CustomerID</em> values.</p>
</li>
<li>
<p>Handled missing categorical values by filling <em>Gender</em> with <em>NotDisclosed</em> and <em>Country</em> and <em>PreferredDevice</em> with <em>Unknown</em>. Handled missing numerical values by filling <em>Age</em> with median age and <em>TotalPurchase</em> with zero.</p>
</li>
<li>
<p>Converted <em>SignupDate</em> and <em>LastLogin</em> columns to datetime format and resolved missing values through logical cross-filling. Converted numerical columns to the correct data types for proper analysis.</p>
</li>
<li>
<p>Removed numeric characters from the <em>Name</em>  column to correct invalid name entries. Replaced negative values in <em>Age</em> with the median age and corrected negative <em>TotalPurchase</em> values by setting them to zero.</p>
</li>
<li>
<p>Standardized categorical columns by fixing common spelling errors and removed duplicate records before saving the cleaned dataset as a new CSV file.</p>
</li>
</ul>
<p><a href="https://github.com/riti2043/MARVEL-Report-02/blob/master/TASK_3.ipynb">Link to notebook</a></p>
<h2 id="task-4--anomaly-detection">TASK 4 : Anomaly Detection</h2>
<p><strong>Anomaly detection</strong> is the process of identifying data points or patterns that significantly deviate from normal behavior. It is commonly used to uncover rare, unusual, or suspicious activities hidden within large datasets.</p>
<p><a href="https://github.com/riti2043/Concepts_for_level2/blob/master/Anomaly%20detection.md">What I learnt about Anomaly Detection</a></p>
<h3 id="dataset-used">Dataset Used</h3>
<p>The dataset contains user activity logs from an internal system, including login behavior, data usage, file downloads, timestamps, and remote access indicators.</p>
<hr>
<h3 id="process-followed-1">Process Followed</h3>
<ul>
<li>
<p>Loaded and explored the dataset to understand structure, ranges, and user behavior</p>
</li>
<li>
<p>Analyzed feature distributions and correlations to identify normal activity trends</p>
</li>
<li>
<p>Visualized data access and file download patterns using histograms and scatter plots</p>
</li>
<li>
<p>Applied statistical methods (Z-score and IQR) to detect extreme deviations</p>
</li>
<li>
<p>Applied unsupervised machine learning (Isolation Forest) after feature scaling</p>
</li>
<li>
<p>Compared anomalies detected by different methods to reduce false positives</p>
</li>
<li>
<p>Identified top suspicious user sessions using multi-feature and multi-model evidence</p>
</li>
</ul>
<p><a href="https://github.com/riti2043/MARVEL-Report-02/blob/master/TASK_4.ipynb">Link to notebook</a></p>
<p><img src="https://github.com/riti2043/images-for-articles/blob/main/TASK_4.jpeg?raw=true" alt=""></p>
<h2 id="task-5--logistic-regression-from-scratch">TASK 5 : Logistic Regression from Scratch</h2>
<p>Logistic Regression is a supervised machine learning algorithm used for <strong>binary classification</strong> problems. It models the probability of a data point belonging to a particular class using the <strong>sigmoid function</strong>, which maps values between 0 and 1. Based on a decision threshold, the model assigns class labels such as 0 or 1.</p>
<p>Logistic Regression is used for <strong>binary classification</strong> when both <strong>probability estimation</strong> and <strong>model interpretability</strong> are important and performs well when the relationship between features and the target is approximately linear.</p>
<p>Logistic Regression requires <strong>feature scaling</strong> and is sensitive to <strong>class imbalance</strong>, making it commonly used as a <strong>baseline model</strong> for comparison.</p>
<p><img src="https://github.com/riti2043/images-for-articles/blob/main/TASK_5.jpeg?raw=true" alt=""></p>
<h3 id="dataset-used-">Dataset used :</h3>
<p>The model was trained and evaluated using the <strong>Framingham Heart Disease dataset</strong>, which contains patient demographic details, clinical measurements, and lifestyle factors to predict the <strong>10-year risk of coronary heart disease (TenYearCHD)</strong>.</p>
<h3 id="process-followed-2">Process followed:</h3>
<ul>
<li>Loaded the dataset and explored its structure, data types, and missing values.</li>
<li>Handled missing categorical values (<em>education</em>, <em>BPMeds</em>) using <strong>mode imputation</strong>.Handled missing numerical values (<em>cigsPerDay</em>, <em>totChol</em>, BMI_, <em>heartRate</em>, <em>glucose</em>) using <strong>mean imputation</strong>.</li>
<li>Verified that the dataset contained <strong>no duplicate records</strong>.</li>
<li>Separated the dataset into <strong>features (X)</strong> and <strong>target labels (y)</strong>.Split the data into training and test sets using <strong>stratified sampling</strong> to preserve class balance.</li>
<li>Normalized feature values using <strong>standardization</strong> to ensure stable and faster convergence.</li>
<li>Implemented <strong>Logistic Regression from scratch</strong> using gradient descent and sigmoid activation.</li>
<li>Trained a <strong>Logistic Regression model using scikit-learn</strong> for comparison</li>
<li>Evaluated both models using <strong>accuracy, precision, recall, and F1-score</strong>.Measured and compared the <strong>training time</strong> of both implementations.</li>
<li>Visualized and compared the <strong>decision boundaries</strong> of both models using PCA for 2D representation.</li>
</ul>
<p><a href="https://github.com/riti2043/MARVEL-Report-02/blob/master/TASK_5.ipynb">Link to notebook</a></p>
<h2 id="task-6--battle-test-your-model---support-vector-machines">TASK 6 : Battle-Test Your Model - Support Vector Machines</h2>
<h3 id="dataset-used-1">Dataset used:</h3>
<p>A multivariate collection of chemical wine profiles. The data was processed into a binary classification format to distinguish high-quality “Good” wines from standard “Not Good” bottles.</p>
<h3 id="process-followed-3">Process followed:</h3>
<ul>
<li>
<p>Cleaned the dataset by handling missing values and duplicates, and performed a binary transformation on the quality labels to create target classes.</p>
</li>
<li>
<p>Applied <code>StandardScaler</code> to the chemical features, ensuring all variables contributed equally to the SVM’s distance-based calculations.</p>
</li>
<li>
<p>Developed a Support Vector Machine (SVM) classifier using an RBF kernel and established a baseline performance using the clean dataset.</p>
</li>
<li>
<p>Validated the baseline model using a confusion matrix, identifying a high Recall strategy (<strong>86.5%</strong>) with a specific breakdown of 180 TN, 55 FP, 5 FN, and 32 TP.</p>
</li>
<li>
<p>Conducted a controlled experiment by adding random noise to the feature set, gradually increasing the standard deviation (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>σ</mi></mrow><annotation encoding="application/x-tex">\sigma</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">σ</span></span></span></span></span>) from <strong>0.01</strong> to <strong>0.50</strong>.</p>
</li>
<li>
<p>Retrained the model at each noise increment and systematically logged changes in <strong>Accuracy</strong>, <strong>Precision</strong>, <strong>Recall</strong>, and <strong>Macro F1-Score</strong>.</p>
</li>
<li>
<p>Identified <strong><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>σ</mi><mo>=</mo><mn>0.02</mn></mrow><annotation encoding="application/x-tex">\sigma = 0.02</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">σ</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.64444em; vertical-align: 0em;"></span><span class="mord">0.02</span></span></span></span></span></strong> as the critical threshold where the model’s discriminative power collapsed, marked by a sharp drop in Recall and F1-score.</p>
</li>
<li>
<p>Concluded that beyond the breakdown point, the noise-to-signal ratio became too high, causing the SVM to lose sensitivity and default to majority-class predictions.</p>
</li>
</ul>
<p><img src="https://github.com/riti2043/images-for-articles/blob/main/TASK_6.jpeg?raw=true" alt=""></p>
<h2 id="task-7-fairness-meets-functionality">TASK 7: Fairness Meets Functionality</h2>
<h3 id="dataset-used--1">Dataset Used :</h3>
<p>The <strong>Utrecht Fairness Recruitment Dataset</strong> was utilized, containing candidate profiles with features such as gender, age, education level, and years of experience. The target variable indicates the final hiring decision.</p>
<h3 id="process-followed-4">Process followed:</h3>
<ul>
<li>
<p>The dataset was filtered to focus on specific predictors, including sensitive demographic attributes like Age and Gender alongside objective performance metrics like test results.</p>
</li>
<li>
<p>Continuous age values were transformed into discrete categories (&lt;25, 25-35, &gt;35) to align the data with the categorical requirements of the ID3 algorithm.</p>
</li>
<li>
<p>A custom Decision Tree structure was built from scratch, utilizing a Node class to store values, child branches, and recursive pointers.</p>
</li>
<li>
<p>An Entropy function was developed to measure data impurity and determine the mathematical basis for splits.</p>
</li>
<li>
<p>An optimization function was created to identify the feature that maximized information gain at each node, ensuring the most influential features were prioritized.</p>
</li>
<li>
<p>The tree was constructed using the ID3 algorithm, incorporating base cases for pure nodes and scenarios where features were exhausted.</p>
</li>
<li>
<p>A traversal function was implemented to navigate the generated tree by matching sample features to the specific values of child nodes.</p>
</li>
<li>
<p>The model’s predictions were audited by slicing the results by Gender and Age groups to generate individual performance metrics for each population segment.</p>
</li>
</ul>
<p><a href="https://github.com/riti2043/MARVEL-Report-02/blob/master/TASK_7.ipynb">Link to notebook</a></p>
<p><img src="https://github.com/riti2043/images-for-articles/blob/main/graphviz%20(2).png?raw=true" alt="graphviz (2).png"></p>
<p>The metrics reveal a lack of <strong>Demographic Parity</strong>, as hiring decisions are heavily dependent on both age and gender; the model shows a severe bias against the <strong>&gt;35</strong> age group (17% recall) and <strong>females</strong>, who face a much lower F1-score (0.62) compared to males (0.76). <strong>Equal Opportunity</strong> is not achieved because qualified candidates from these demographics are significantly less likely to be “hired” than their younger, male counterparts.<br>
Hence the tree’s reliance on sensitive attributes as primary splits fails ethically, as it utilizes demographic proxies to create inconsistent and discriminatory hiring criteria across all groups.</p>
<h2 id="task-8--knn-with-ablation-study">TASK 8 : KNN with Ablation Study</h2>
<p>In KNN algorithm , the majority class label determines the class label of a new data point among its k nearest neighbors.<br>
KNN is a proximity-based learning and hence the fundamental concept is - <strong>closeness dictates similarity</strong>.<br>
This <strong>closeness</strong> is determined by a distance metric, commonly Euclidean or Manhattan distance.<br>
<img src="https://github.com/riti2043/images-for-articles/blob/main/Task_82.jpeg?raw=true" alt="euclid_dis"></p>
<h3 id="how-knn-works">How KNN works</h3>
<ul>
<li>Given a new data point, calculate its distance from all other data points in the dataset.</li>
<li>Get the closest K points.</li>
<li>Regression : Get the average of their values.</li>
<li>Classification : Get the label with the majority vote.</li>
</ul>
<h3 id="dataset-used--2">Dataset Used :</h3>
<p>The <strong>Breast Cancer Wisconsin (Diagnostic) Dataset</strong> was used, containing 30 clinical features. The goal is to classify tumors as either <strong>Malignant (1)</strong> or <strong>Benign (0)</strong>.</p>
<h3 id="process-followed-5">Process followed:</h3>
<ul>
<li>
<p>Loaded the dataset, dropped unique identifiers, and encoded the diagnosis target into a binary format (M=1, B=0).</p>
</li>
<li>
<p>Applied <strong>StandardScaler</strong> to normalize the 30 features, ensuring that larger-scale measurements (like area) did not overshadow smaller-scale metrics (like smoothness).</p>
</li>
<li>
<p>Established a performance benchmark using all 30 features, achieving a baseline accuracy of <strong>94.74%</strong> with an F1-score of <strong>0.9302</strong>.</p>
</li>
<li>
<p>Removed one feature at a time and retrained the model over 30 iterations to measure the resulting impact on Accuracy, Precision, Recall, and F1-score.</p>
</li>
<li>
<p>Analyzed the “Accuracy Drop” for each iteration, discovering that removing certain features actually <strong>improved</strong> model performance (negative accuracy drop).</p>
</li>
<li>
<p>Identified and removed the top 10 “noisy” features , reducing the feature set to the 20 most informative variables.</p>
</li>
<li>
<p>Retrained the optimized model, resulting in a performance lift from <strong>94.74% to 97.37%</strong> accuracy.</p>
</li>
<li>
<p>Implemented the <strong>KNN algorithm from scratch</strong>  to verify the underlying logic.</p>
</li>
</ul>
<p><a href="https://github.com/riti2043/MARVEL-Report-02/blob/master/TASK_8.ipynb">Link to notebook</a></p>
<p><img src="https://github.com/riti2043/images-for-articles/blob/main/TASK_8.jpeg?raw=true" alt="METRICS"></p>
<h2 id="task-9--evaluation-metrics---pick-the-best-performer">TASK 9 : Evaluation Metrics - Pick the Best Performer!</h2>
<p><strong>Pickle files</strong> are commonly used in machine learning to save trained models so they can be reused later without training them again.<br>
<strong>Joblib</strong> is a Python library designed for efficiently loading and saving such models, and it was used here to load all the pretrained <code>.pkl</code> files provided for evaluation.</p>
<p>For this task, the <strong>Iris dataset</strong> was chosen as it is a simple, balanced multiclass classification dataset suitable for comparing different models.</p>
<p><a href="https://github.com/riti2043/Concepts_for_level2/blob/master/Evaluation%20metrics.md">What I learnt about Evaluation Metrics</a></p>
<h3 id="process-followed-6"><strong>Process Followed</strong>:</h3>
<ol>
<li>
<p>Loaded the Iris dataset and separated it into features and target labels.</p>
</li>
<li>
<p>Verified the input shape to ensure it matched what the pretrained models expect.</p>
</li>
<li>
<p>Loaded all five pretrained models using joblib.</p>
</li>
<li>
<p>Checked the class labels of each model to confirm label compatibility.</p>
</li>
<li>
<p>Generated predictions for each model on the dataset.</p>
</li>
<li>
<p>Evaluated performance using accuracy, precision, recall, and F1-score (macro-averaged).</p>
</li>
<li>
<p>Compared all models by summarizing the results in a single comparison table.</p>
</li>
</ol>
<p><a href="https://github.com/riti2043/MARVEL-Report-02/blob/master/TASK_9.ipynb">Link to notebook</a></p>
<h3 id="model-comparison-and-conclusion"><strong>Model Comparison and Conclusion</strong></h3>
<p>Decision Tree showed slightly lower performance due to rigid splits and minor confusion between overlapping classes. KNN performed well but was sensitive to local neighborhood variations. Random Forest provided stable results through ensemble learning but did not outperform the top models. Logistic Regression and SVM achieved the highest and identical performance metrics.</p>
<p>Although Logistic Regression and SVM produced similar results, <strong>SVM was chosen as the best-performing model</strong> due to its <strong>margin maximization principle</strong>, which constructs a more robust decision boundary and offers better generalization when class boundaries overlap.</p>
<p><img src="https://github.com/riti2043/images-for-articles/blob/main/Task9.jpeg?raw=true" alt="models"></p>


# Part B — Classical Track (Shuvo)

তোমার অংশের গল্প এক লাইনে: **"পুরোনো ধাঁচের model দিয়ে কতদূর যাওয়া যায়, কত দ্রুত, আর কোথায় ভেঙে পড়ে।"**

## ফাইল

| ফাইল | কী করে |
|---|---|
| `src/features_tfidf.py` | TF-IDF feature — word (1,2)-gram + char_wb (3,5)-gram |
| `src/baselines.py` | LogReg, Multinomial NB, Linear SVM — train ও evaluate |

চালানো (project root থেকে, `src/split.py` আগে চলতে হবে):
```bash
python src/baselines.py
```

## ফলাফল

| Condition | n | accuracy | metric |
|---|---|---|---|
| E1 Test-Clean | 100 | 0.910 | macro-F1 **0.900** |
| E4 Test-CodeMix (বাংলা) | 900 | 0.716 | macro-F1 **0.705** |
| E5 Test-CrossDataset (Lakera) | 1000 | 0.990 | recall 0.990 |
| FP-test (PromptBench) | 4150 | 0.021 | **FPR 0.979** |

(উপরের সংখ্যা Logistic Regression-এর; তিন model-ই প্রায় একই।)

## 🔴 সবচেয়ে গুরুত্বপূর্ণ ব্যাপার — এটা viva-র আসল প্রশ্ন

E5-এ recall **০.৯৯** দেখে "দারুণ generalize করছে" বলা যাবে **না**। পাশের
ঘরেই FPR **০.৯৭৯** — মানে PromptBench-এর ৪,১৫০টা **নিরীহ** task prompt-এর
৯৮%-কে model "আক্রমণ" বলছে।

কারণটা ধরা পড়েছে predicted-rate দেখে:

| set | model যা বলছে | সত্যি |
|---|---|---|
| Test-Clean | ৩৫% injection | ৩৪% ✅ ঠিকঠাক |
| PromptBench | **৯৮%** injection | ০% ❌ |
| CodeMix | ৩১% injection | ৫০% ❌ কম ধরছে |

অর্থাৎ model globally "সব কিছুকে আক্রমণ" বলছে না (Test-Clean-এ তো ঠিকই আছে)।
সে শিখেছে **"ইংরেজি আদেশের সুর = আক্রমণ"**। Lakera-র আক্রমণগুলো আদেশের সুরে
লেখা → ঠিক ধরছে। কিন্তু PromptBench-এর task prompt-ও আদেশের সুরে
(`"Examine the sentence and decide if its grammar is 'Acceptable'"`) → ভুল ধরছে।

**Model "model-কে দেওয়া আদেশ" আর "model-কে বিপথে নেওয়ার আদেশ" — এই দুটোর
পার্থক্য শেখেনি।** E5 আর FP-test একসাথে না দেখালে এই কথাটা ধরাই পড়ত না।

## দুই নম্বর ফলাফল — বাংলায় ভেঙে পড়ছে

E1-এ ০.৯০০ → E4-এ ০.৭০৫। আর বাংলায় model মাত্র ৩১% সারিকে injection বলছে,
যদিও আসলে ৫০%। **ইংরেজিতে প্রশিক্ষিত model বাংলা আক্রমণ মিস করছে** — এটাই
RQ3, আর H3-র পক্ষে প্রমাণ।

## যা জানতে চাইবে

**ক) char n-gram কেন?** শুধু word n-gram হলে obfuscation (`1gn0r3`,
`i g n o r e`) পুরো token ভেঙে দিত। char_wb (3,5) তাতেও টিকে থাকে।

**খ) `class_weight='balanced'` কেন?** ডেটা ৭৪% Safe। না দিলে model সব
"Safe" বলেই ৭৪% accuracy পেত, একটাও আক্রমণ না ধরে।

**গ) Naive Bayes-এ class_weight নেই কেন?** sklearn-এর MultinomialNB এটা
support করে না — `class_prior` দিয়ে করা যায়। এটা জেনে রাখা ভালো।

**ঘ) Train মাত্র ৪৬২ সারি কেন?** Lakera আর PromptBench দুটোই held-out test,
তাই train-এ শুধু deepset। এটা একটা সীমাবদ্ধতা, paper-এ লিখতে হবে।

**ঙ) Latency** — `baseline_results.csv`-এ `ms_per_row` কলাম আছে (E7)।
তিনটা model-ই ০.১ সেকেন্ডের কমে train হয় — transformer-এর সাথে তুলনার জন্য।

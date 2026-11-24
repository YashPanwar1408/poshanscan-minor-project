import { usePortalPrefs, type PortalLanguage } from "@/lib/portal-prefs";
import type { ReactNode } from "react";

// Keys are the English source copy used by the interface. Grouping by product
// area keeps common UI, worker flows, screening copy and supervisor modules
// reviewable without embedding language branches in pages.
const common = {
  All: "सभी",
  Cancel: "रद्द करें",
  Close: "बंद करें",
  Copy: "कॉपी करें",
  Date: "तारीख",
  Dismiss: "हटाएँ",
  Documentation: "दस्तावेज़",
  English: "English",
  Female: "महिला",
  Male: "पुरुष",
  Other: "अन्य",
  Next: "अगला",
  No: "नहीं",
  Page: "पृष्ठ",
  Previous: "पिछला",
  Refresh: "रीफ़्रेश करें",
  Retake: "फिर से लें",
  Retry: "दोबारा प्रयास करें",
  "Retry Connection": "कनेक्शन फिर से आज़माएँ",
  "Retry Download": "डाउनलोड फिर से आज़माएँ",
  "Go home": "होम पर जाएँ",
  "Back to home": "होम पर वापस जाएँ",
  "Page not found": "पृष्ठ नहीं मिला",
  "The page you're looking for doesn't exist or has been moved.":
    "आप जो पृष्ठ ढूँढ़ रहे हैं, वह मौजूद नहीं है या कहीं और ले जाया गया है।",
  "This page didn't load": "यह पृष्ठ लोड नहीं हुआ",
  "Something went wrong on our end. You can try refreshing or head back home.":
    "कुछ गड़बड़ हुई। आप रीफ़्रेश कर सकते हैं या होम पृष्ठ पर लौट सकते हैं।",
  "Try again": "फिर से प्रयास करें",
  "Toggle Columns": "कॉलम दिखाएँ या छिपाएँ",
  "Loading…": "लोड हो रहा है…",
  "Searching…": "खोज जारी है…",
  "Generating CSV…": "CSV बनाया जा रहा है…",
  "Signing in…": "साइन इन हो रहा है…",
  "for shortcut sheet": "शॉर्टकट सूची के लिए",
  "to navigate": "जाने के लिए",
  "to select": "चुनने के लिए",
  "in one place.": "एक ही जगह पर।",
  Language: "भाषा",
  "Text size": "पाठ का आकार",
  "High contrast": "उच्च कंट्रास्ट",
  "Decrease text size": "पाठ छोटा करें",
  "Default text size": "डिफ़ॉल्ट पाठ आकार",
  "Increase text size": "पाठ बड़ा करें",
  "Switch to light mode": "लाइट मोड पर जाएँ",
  "Switch to dark mode": "डार्क मोड पर जाएँ",
  "Switch to light theme": "लाइट थीम पर जाएँ",
  "Switch to dark theme": "डार्क थीम पर जाएँ",
  "Log out": "लॉग आउट",
  "Sign Out of Portal": "पोर्टल से साइन आउट करें",
  "Skip to main content": "मुख्य सामग्री पर जाएँ",
  "Close Profile": "प्रोफ़ाइल बंद करें",
  "Main Navigation": "मुख्य नेविगेशन",
  "Next page": "अगला पृष्ठ",
  "Previous page": "पिछला पृष्ठ",
  "Close zoomed image": "ज़ूम की गई तस्वीर बंद करें",
} satisfies Record<string, string>;

const auth = {
  "FIELD WORKER ACCESS": "फील्ड कार्यकर्ता प्रवेश",
  "PROGRAM SUPERVISOR ACCESS": "कार्यक्रम पर्यवेक्षक प्रवेश",
  "Log in as a worker": "कार्यकर्ता के रूप में लॉग इन करें",
  "Log in as a supervisor": "पर्यवेक्षक के रूप में लॉग इन करें",
  "Start guided screenings, capture photos, and follow up with children.":
    "निर्देशित जाँच शुरू करें, तस्वीरें लें और बच्चों की आगे की देखभाल करें।",
  "Review program progress, spot follow-up needs, and support your field team.":
    "कार्यक्रम की प्रगति देखें, आगे की देखभाल की ज़रूरत पहचानें और अपनी फील्ड टीम की मदद करें।",
  "Open worker workspace": "कार्यकर्ता का कार्यक्षेत्र खोलें",
  "Open supervisor dashboard": "पर्यवेक्षक डैशबोर्ड खोलें",
  "Worker access is provided by your program administrator.":
    "कार्यकर्ता की पहुँच कार्यक्रम व्यवस्थापक देता है।",
  "Supervisor access is managed by your program administrator.":
    "पर्यवेक्षक की पहुँच कार्यक्रम व्यवस्थापक संभालता है।",
  "Capture clearly. Screen consistently. Support better care.":
    "साफ़ तस्वीर लें। लगातार जाँच करें। बेहतर देखभाल में मदद करें।",
  "Use guided photo capture to record reliable screening information in the field.":
    "फील्ड में भरोसेमंद जाँच जानकारी दर्ज करने के लिए निर्देशित तस्वीर लें।",
  "See program progress. Coordinate timely follow-up.":
    "कार्यक्रम की प्रगति देखें। समय पर आगे की देखभाल का समन्वय करें।",
  "Review screening trends and help field teams connect children with the right care.":
    "जाँच के रुझान देखें और बच्चों को उचित देखभाल दिलाने में फील्ड टीमों की मदद करें।",
  "Start a screening": "जाँच शुरू करें",
  "Review program activity": "कार्यक्रम की गतिविधि देखें",
  "Supervisor login": "पर्यवेक्षक लॉग इन",
  "Worker login": "कार्यकर्ता लॉग इन",
  "Create a worker account": "कार्यकर्ता खाता बनाएँ",
  "Create your account to start field screenings.": "फील्ड जाँच शुरू करने के लिए अपना खाता बनाएँ।",
  "Create worker account": "कार्यकर्ता खाता बनाएँ",
  "Full name": "पूरा नाम",
  "Enter your full name": "अपना पूरा नाम लिखें",
  "Email address": "ईमेल पता",
  "Enter your email (optional)": "अपना ईमेल लिखें (वैकल्पिक)",
  "Village or field area": "गाँव या कार्य क्षेत्र",
  "Enter your working area": "अपना कार्य क्षेत्र लिखें",
  "Phone number": "फ़ोन नंबर",
  "Enter your phone number": "अपना फ़ोन नंबर लिखें",
  Password: "पासवर्ड",
  "Enter your password": "अपना पासवर्ड लिखें",
  "Confirm password": "पासवर्ड की पुष्टि करें",
  "Re-enter your password": "पासवर्ड फिर से लिखें",
  "Already have an account?": "क्या आपका पहले से खाता है?",
  "New worker?": "नए कार्यकर्ता हैं?",
  "Sign up": "खाता बनाएँ",
  "Sign in": "साइन इन करें",
  "Looking for the other workspace?": "दूसरा कार्यक्षेत्र ढूँढ़ रहे हैं?",
  "Please check your details.": "कृपया अपनी जानकारी जाँचें।",
  "Phone number or admin ID": "फ़ोन नंबर या व्यवस्थापक आईडी",
  "Phone number or admin ID, plus your password.": "फ़ोन नंबर या व्यवस्थापक आईडी और आपका पासवर्ड।",
  "Stay signed in": "साइन इन बने रहें",
  "Your session has expired. Please sign in again.":
    "आपका सत्र समाप्त हो गया है। कृपया फिर से साइन इन करें।",
  "Your session will expire in less than 60 seconds.":
    "आपका सत्र 60 सेकंड से कम समय में समाप्त हो जाएगा।",
  "Session Expiring Soon": "सत्र जल्द समाप्त होगा",
  "e.g. 9000000000 or admin": "जैसे 9000000000 या admin",
  "Worker profile": "कार्यकर्ता प्रोफ़ाइल",
  "This account is not a {role} account. Please use the {userRole} login.":
    "यह {role} खाता नहीं है। कृपया {userRole} लॉग इन का उपयोग करें।",
} satisfies Record<string, string>;

const worker = {
  "Who is being screened?": "किसकी जाँच की जा रही है?",
  "Choose a category to continue. No category is selected by default.":
    "आगे बढ़ने के लिए श्रेणी चुनें। पहले से कोई श्रेणी नहीं चुनी गई है।",
  "Screening category": "जाँच की श्रेणी",
  "New screening": "नई जाँच",
  "Search children": "बच्चों को खोजें",
  "Register new": "नया पंजीकरण करें",
  "Register new adult": "वयस्क का नया पंजीकरण करें",
  "Register new child": "बच्चे का नया पंजीकरण करें",
  Female: "महिला",
  Male: "पुरुष",
  "Capture photo": "तस्वीर लें",
  "Upload photo": "तस्वीर अपलोड करें",
  "Arm Scan Capture Photo": "बाँह की जाँच की तस्वीर लें",
  "Get a clear reading": "स्पष्ट माप प्राप्त करें",
  "CAPTURE CHECKLIST": "तस्वीर लेने की जाँच सूची",
  "Place the printed marker flat on the middle of the upper arm.":
    "छपे हुए मार्कर को ऊपरी बाँह के बीच में सपाट रखें।",
  "Keep the arm relaxed and the marker facing the camera.":
    "बाँह को ढीला रखें और मार्कर को कैमरे की ओर रखें।",
  "Ensure both arm and marker fit inside the square guide.":
    "ध्यान रखें कि बाँह और मार्कर दोनों चौकोर गाइड के भीतर हों।",
  "Use even lighting and avoid blur or shadows.":
    "समान रोशनी रखें और धुंधली तस्वीर या छाया से बचें।",
  "Keep arm and marker inside the guide": "बाँह और मार्कर को गाइड के भीतर रखें",
  "Perpendicular angle (90°), mid-arm marker level, natural indirect illumination.":
    "कैमरे को बाँह से 90° पर रखें, मार्कर बाँह के मध्य में हो और प्राकृतिक अप्रत्यक्ष रोशनी का उपयोग करें।",
  "Usually 50 mm. Enter the actual size if your print differs.":
    "आमतौर पर 50 mm। यदि प्रिंट का आकार अलग है, तो वास्तविक माप लिखें।",
  "Measured marker size (mm)": "मार्कर का मापा हुआ आकार (mm)",
  "ArUco 50mm locked": "ArUco 50 mm तय है",
  "ArUco 50 mm Reference": "ArUco 50 mm संदर्भ",
  "Download ArUco Marker": "ArUco मार्कर डाउनलोड करें",
  "Your scan is safely saved on this device. It will upload when your connection returns.":
    "आपकी जाँच इस डिवाइस पर सुरक्षित है। कनेक्शन लौटने पर यह अपलोड हो जाएगी।",
  "Saved for sync": "सिंक के लिए सुरक्षित",
  "An offline scan was not accepted.": "ऑफ़लाइन जाँच स्वीकार नहीं हुई।",
  "The scan remains stored on this device and stays pending until the server accepts it.":
    "जाँच इस डिवाइस पर सुरक्षित है और सर्वर की स्वीकृति तक लंबित रहेगी।",
  "Retry rejected scans": "अस्वीकृत जाँच फिर से भेजें",
  "pending sync": "सिंक लंबित",
  "Please choose an image file.": "कृपया तस्वीर फ़ाइल चुनें।",
  "Please capture or upload a photo first.": "पहले तस्वीर लें या अपलोड करें।",
  "Please choose whether this screening is for a child or an adult.":
    "कृपया चुनें कि यह जाँच बच्चे के लिए है या वयस्क के लिए।",
  "Enter a valid marker size before submitting.": "भेजने से पहले मार्कर का सही आकार लिखें।",
  "Could not save your photo offline. Please try again.":
    "तस्वीर ऑफ़लाइन सुरक्षित नहीं हो सकी। कृपया फिर से प्रयास करें।",
  "Your session has expired. Please log in again.":
    "आपका सत्र समाप्त हो गया है। कृपया फिर से लॉग इन करें।",
  "Back to screening": "जाँच पर वापस जाएँ",
  "All adults": "सभी वयस्क",
  "All children": "सभी बच्चे",
  "View all": "सभी देखें",
  "Searching…": "खोज जारी है…",
  "No workers found": "कोई कार्यकर्ता नहीं मिला",
  "No workers match your filter criteria or search query.":
    "आपके फ़िल्टर या खोज से कोई कार्यकर्ता मेल नहीं खाता।",
  "Registered Child Record": "पंजीकृत बच्चे का रिकॉर्ड",
  "Close zoomed image": "ज़ूम की गई तस्वीर बंद करें",
} satisfies Record<string, string>;

const screening = {
  "MUAC assessment summary": "MUAC जाँच का सारांश",
  "POSHANSCAN SCREENING REPORT": "POSHANSCAN जाँच रिपोर्ट",
  "Screening report": "जाँच रिपोर्ट",
  "Screening details": "जाँच का विवरण",
  "SCREENING RESULT": "जाँच का परिणाम",
  "SCREENING METADATA": "जाँच का मेटाडेटा",
  "SCREENING CHECKLIST": "जाँच सूची",
  "CAPTURED PHOTO": "ली गई तस्वीर",
  "MEASUREMENT SCALE": "माप का पैमाना",
  "Received MUAC": "प्राप्त MUAC",
  "Estimated MUAC of": "अनुमानित MUAC",
  "MUAC:": "MUAC:",
  "mm MUAC": "mm MUAC",
  "MUAC ≈ π · k · width. Evaluated against Shakir tape measurements (k ≈ 1.22–1.35).":
    "MUAC ≈ π · k · चौड़ाई। इसका मूल्यांकन शाकिर टेप माप (k ≈ 1.22–1.35) से किया गया।",
  "Adjust Circumference (MUAC)": "परिधि (MUAC) बदलें",
  "MUAC Measurement Scale (60–180 mm)": "MUAC माप पैमाना (60–180 mm)",
  "WHO MUAC Band Explorer": "WHO MUAC श्रेणी खोजक",
  "WHO Screening Band": "WHO जाँच श्रेणी",
  "WHO Triage Classification": "WHO प्राथमिकता वर्गीकरण",
  "WHO Screening Bands Explanation": "WHO जाँच श्रेणियों की जानकारी",
  "WHO 2006 Child Growth Standards": "WHO 2006 बाल विकास मानक",
  SAM: "SAM",
  MAM: "MAM",
  Normal: "सामान्य",
  "Normal status. No clinical flags requiring urgent triage.":
    "सामान्य स्थिति। तत्काल जाँच की आवश्यकता वाला कोई चिकित्सकीय संकेत नहीं।",
  "Adequate nutrition status; routine growth monitoring.":
    "पोषण स्थिति पर्याप्त है; नियमित वृद्धि की निगरानी करें।",
  "Moderate Acute Malnutrition; supplementary feeding support.":
    "मध्यम तीव्र कुपोषण; अतिरिक्त पोषण सहायता की आवश्यकता।",
  "Severe Acute Malnutrition; urgent clinical referral.":
    "गंभीर तीव्र कुपोषण; तुरंत स्वास्थ्य केंद्र भेजें।",
  "Recommended next step": "अगला सुझाया गया कदम",
  "Recommended Next Step": "अगला सुझाया गया कदम",
  "Recommended Action:": "सुझाई गई कार्रवाई:",
  "Review the measured value, applicable scale, and recommended follow-up.":
    "माप, लागू पैमाना और सुझाई गई आगे की देखभाल देखें।",
  "No elevated screening risk indicated by this scale.":
    "इस पैमाने पर जाँच का जोखिम बढ़ा हुआ नहीं दिखता।",
  "Screening aid — not a diagnosis. Always confirm with a trained health worker.":
    "यह जाँच में सहायता का साधन है, निदान नहीं। प्रशिक्षित स्वास्थ्यकर्मी से पुष्टि करें।",
  "Screening aid, not a diagnosis": "जाँच में सहायता का साधन, निदान नहीं",
  "Low confidence — repeat the scan and confirm with a standard MUAC tape.":
    "विश्वास कम है — जाँच दोहराएँ और मानक MUAC टेप से पुष्टि करें।",
  "Low Conf": "कम विश्वास",
  "Low confidence": "कम विश्वास",
  "Estimated MUAC": "अनुमानित MUAC",
  "Screen again": "फिर से जाँचें",
  "Scan another child": "किसी अन्य बच्चे की जाँच करें",
  "Choose another person": "किसी अन्य व्यक्ति को चुनें",
  "SAM (<115)": "SAM (<115)",
  "SAM (<": "SAM (<",
  "SAM Cases": "SAM मामले",
  "MAM Cases": "MAM मामले",
  "SAM CASES": "SAM मामले",
  "SAM Cases in Period": "अवधि में SAM मामले",
  "SAM Rate": "SAM दर",
  "SAM Rate:": "SAM दर:",
  "SAM Share": "SAM का हिस्सा",
  "MAM: 115 to < 125 mm": "MAM: 115 से < 125 mm",
  "SAM: < 115 mm": "SAM: < 115 mm",
  "Normal: ≥ 125 mm": "सामान्य: ≥ 125 mm",
  "115 mm (SAM cutoff)": "115 mm (SAM सीमा)",
  "125 mm (MAM cutoff)": "125 mm (MAM सीमा)",
  "Adult thresholds: Normal ≥230 mm, MAM 210–<230 mm, SAM <210 mm.":
    "वयस्क सीमा: सामान्य ≥230 mm, MAM 210–<230 mm, SAM <210 mm।",
  "Instant categorization: Normal (≥125 mm), MAM (115–<125 mm), SAM (<115 mm).":
    "तुरंत वर्गीकरण: सामान्य (≥125 mm), MAM (115–<125 mm), SAM (<115 mm)।",
  "Clinical & Programme Disclaimer": "चिकित्सकीय और कार्यक्रम संबंधी सूचना",
  "Clinical & Regulatory Notice:": "चिकित्सकीय और नियामक सूचना:",
  "Note:": "ध्यान दें:",
  "PoshanScan is a digital screening aid. Any child flagged as SAM or MAM must be clinically verified with standard Shakir tape and assessed for oedema.":
    "PoshanScan डिजिटल जाँच में सहायता करता है। SAM या MAM संकेत वाले प्रत्येक बच्चे की मानक शाकिर टेप से पुष्टि करें और सूजन की जाँच कराएँ।",
  "PoshanScan is a digital screening aid, never a medical diagnosis. Always confirm results with a trained frontline healthcare provider.":
    "PoshanScan डिजिटल जाँच में सहायता करता है, यह चिकित्सकीय निदान नहीं है। परिणाम की पुष्टि प्रशिक्षित स्वास्थ्यकर्मी से करें।",
  "PoshanScan is a screening aid, not a medical diagnosis. Always confirm abnormal or low-confidence results with a trained health worker.":
    "PoshanScan जाँच में सहायता करता है, यह चिकित्सकीय निदान नहीं है। असामान्य या कम विश्वास वाले परिणाम की पुष्टि प्रशिक्षित स्वास्थ्यकर्मी से करें।",
  "PoshanScan is strictly a screening and triage support tool. It does not provide medical diagnoses, prescribe medical treatments, or replace physical examinations by qualified pediatric or health care personnel. All children identified with Mid-Upper Arm Circumference values in the MAM or SAM ranges must be clinically confirmed with standard Shakir tape measurement and assessed for bilateral pitting oedema before therapeutic nutrition admission.":
    "PoshanScan केवल जाँच और प्राथमिकता तय करने में सहायता करता है। यह चिकित्सकीय निदान या उपचार नहीं देता और योग्य स्वास्थ्यकर्मी की जाँच का विकल्प नहीं है। MAM या SAM सीमा में MUAC वाले प्रत्येक बच्चे का मानक शाकिर टेप से चिकित्सकीय सत्यापन करें और उपचारात्मक पोषण में प्रवेश से पहले दोनों पैरों में दबाने पर रहने वाली सूजन की जाँच करें।",
  "PoshanScan is a digital child-nutrition screening aid developed to assist frontline health workers in measuring Mid-Upper Arm Circumference (MUAC). It connects field assessments with public health supervision to detect wasting earlier and more consistently.":
    "PoshanScan बच्चों के पोषण की डिजिटल जाँच में मदद करता है। यह स्वास्थ्यकर्मियों को मध्य-ऊपरी बाँह की परिधि (MUAC) मापने और समुदाय की जाँच को सार्वजनिक स्वास्थ्य निगरानी से जोड़ने में सहायता देता है।",
  "Screening aid — immediate facility follow-up required.":
    "जाँच में सहायता का साधन; तुरंत स्वास्थ्य केंद्र में आगे की जाँच कराएँ।",
} satisfies Record<string, string>;

const supervisor = {
  Overview: "अवलोकन",
  Trends: "रुझान",
  Flagged: "चिह्नित",
  "Flagged Scans": "चिह्नित जाँच",
  "Flagged Review Queue": "चिह्नित जाँच की समीक्षा सूची",
  "Flagged Reviews": "चिह्नित जाँच की समीक्षाएँ",
  "Situation Room (Overview)": "स्थिति कक्ष (अवलोकन)",
  "Supervisor Portal": "पर्यवेक्षक पोर्टल",
  "Supervisor navigation": "पर्यवेक्षक नेविगेशन",
  "Supervisor Portal Interface Previews": "पर्यवेक्षक पोर्टल के इंटरफ़ेस पूर्वावलोकन",
  "API Inspector": "API निरीक्षक",
  'Click "Run all endpoints" to inspect the API.': "API देखने के लिए “Run all endpoints” चुनें।",
  "Not available in production.": "प्रोडक्शन में उपलब्ध नहीं है।",
  "Dev-only: calls every supervisor endpoint and shows raw responses.":
    "केवल विकास के लिए: सभी पर्यवेक्षक endpoints कॉल करता है और उनके कच्चे जवाब दिखाता है।",
  "Children 6–59m": "6–59 माह के बच्चे",
  Adults: "वयस्क",
  "Age category filter": "आयु श्रेणी फ़िल्टर",
  "Children 6–59 months (Default)": "6–59 माह के बच्चे (डिफ़ॉल्ट)",
  "Filter: Children 6–59 months (Default)": "फ़िल्टर: 6–59 माह के बच्चे (डिफ़ॉल्ट)",
  "Filter: Adults": "फ़िल्टर: वयस्क",
  "Flagged first": "पहले चिह्नित",
  "Newest first": "नवीनतम पहले",
  "Lowest Confidence": "सबसे कम विश्वास",
  "Lowest MUAC": "सबसे कम MUAC",
  "Child Name (A–Z)": "बच्चे का नाम (A–Z)",
  "Mark as Reviewed": "समीक्षा पूरी चिह्नित करें",
  "Flagged for supervisor quality verification.": "पर्यवेक्षक द्वारा गुणवत्ता जाँच के लिए चिह्नित।",
  "Field Worker": "फील्ड कार्यकर्ता",
  "Village / Site": "गाँव / केंद्र",
  "Guardian:": "अभिभावक:",
  "Gender:": "लिंग:",
  "DOB:": "जन्म तिथि:",
  "Reference:": "संदर्भ:",
  "Ref:": "संदर्भ:",
  "Click to zoom photo": "तस्वीर ज़ूम करने के लिए क्लिक करें",
  "Arm photo media attachment unavailable for this scan.":
    "इस जाँच की बाँह वाली तस्वीर उपलब्ध नहीं है।",
  "Choose a flagged scan from the left queue to inspect the visual MUAC scale, model confidence breakdown, and take clinical referral action.":
    "MUAC पैमाना और मॉडल का विश्वास देखने तथा चिकित्सकीय रेफ़रल तय करने के लिए बाईं सूची से चिह्नित जाँच चुनें।",
  "No pending flagged scans require supervisor review for this worker.":
    "इस कार्यकर्ता की कोई लंबित चिह्नित जाँच पर्यवेक्षक की समीक्षा के लिए नहीं है।",
  "No pending scans requiring review.": "समीक्षा के लिए कोई लंबित जाँच नहीं है।",
  "Select a scan to review": "समीक्षा के लिए जाँच चुनें",
  "Select Page (": "पृष्ठ चुनें (",
  "Select Page ({count})": "पृष्ठ चुनें ({count})",
  "Refresh data": "डेटा रीफ़्रेश करें",
  "Updated {elapsed}": "अपडेट किया गया: {elapsed}",
  "Assigned Villages (": "निर्धारित गाँव (",
  "Workers covering this village (": "इस गाँव में कार्यरत लोग (",
  "View All Flagged": "सभी चिह्नित जाँच देखें",
  "Group by:": "इसके अनुसार समूह बनाएँ:",
  "Metric:": "माप:",
  "Mode:": "मोड:",
  "Highest Risk": "सबसे अधिक जोखिम",
  "Highest Volume": "सबसे अधिक संख्या",
  "Name (A-Z)": "नाम (A-Z)",
  "Reset Map View to encompass all coordinates":
    "सभी निर्देशांकों को दिखाने के लिए मानचित्र रीसेट करें",
  "Reset Map View": "मानचित्र दृश्य रीसेट करें",
  "Basemap tiles unavailable offline — location markers still active":
    "ऑफ़लाइन आधार मानचित्र उपलब्ध नहीं है — स्थान चिह्न फिर भी सक्रिय हैं",
  "High risk villages table": "उच्च जोखिम वाले गाँवों की तालिका",
  "Highest-Risk Villages": "सबसे अधिक जोखिम वाले गाँव",
  "Highest-Risk Villages (Min Volume n ≥ 5)": "सबसे अधिक जोखिम वाले गाँव (न्यूनतम संख्या n ≥ 5)",
  "Villages ranked by combined MAM+SAM share (min. volume n≥5)":
    "MAM और SAM के कुल अनुपात के आधार पर गाँवों की सूची (न्यूनतम संख्या n≥5)",
  "Hollow: Low sample (n < 5)": "खोखला चिह्न: नमूना कम (n < 5)",
  "< 5% (Low risk)": "< 5% (कम जोखिम)",
  "5% – 10% (Alert)": "5% – 10% (चेतावनी)",
  "≥ 10% (Critical)": "≥ 10% (गंभीर)",
  "< 10% (Acceptable)": "< 10% (स्वीकार्य)",
  "10% – 15% (Poor)": "10% – 15% (कमज़ोर)",
  "≥ 15% (Critical)": "≥ 15% (गंभीर)",
  "Circle radius scales with screening volume": "वृत्त का आकार जाँच की संख्या के अनुसार है",
  "Band Breakdown": "श्रेणी का विवरण",
  "Band Explorer": "श्रेणियों का विवरण",
  "Risk Classification Distribution": "जोखिम वर्गीकरण का वितरण",
  "GAM Prevalence": "GAM का प्रसार",
  "GAM Rate": "GAM दर",
  "GAM Rate:": "GAM दर:",
  "SAM Prevalence": "SAM का प्रसार",
  "Needs Attention": "ध्यान देने की ज़रूरत",
  "Most Active Workers": "सबसे सक्रिय कार्यकर्ता",
  "Top 5 Workers by Volume": "जाँच संख्या के अनुसार शीर्ष 5 कार्यकर्ता",
  "Top frontline team members by total screening count":
    "कुल जाँच संख्या के अनुसार अग्रिम पंक्ति के शीर्ष सदस्य",
  "Total Field Team": "कुल फील्ड टीम",
  "Active registered staff": "सक्रिय पंजीकृत कर्मचारी",
  "Active (≤24h)": "सक्रिय (≤24 घंटे)",
  "Past week inactive": "पिछले सप्ताह से निष्क्रिय",
  "Dormant (>7d)": "निष्क्रिय (>7 दिन)",
  "Idle (≤7d)": "निष्क्रिय (≤7 दिन)",
  "No active workers reported.": "कोई सक्रिय कार्यकर्ता दर्ज नहीं है।",
  "No designated villages assigned to this worker.": "इस कार्यकर्ता को कोई गाँव निर्धारित नहीं है।",
  "No field workers assigned specifically to this location.":
    "इस स्थान के लिए कोई फील्ड कार्यकर्ता निर्धारित नहीं है।",
  "No locations match your search.": "आपकी खोज से कोई स्थान मेल नहीं खाता।",
  "No village data meeting the volume threshold.":
    "संख्या की तय सीमा पूरी करने वाला गाँव का डेटा नहीं है।",
  "No villages currently meet the high-risk volume threshold.":
    "अभी कोई गाँव उच्च जोखिम की तय संख्या सीमा पूरी नहीं करता।",
  "No trend data available": "रुझान का डेटा उपलब्ध नहीं है",
  "No screening volume recorded.": "जाँच की कोई संख्या दर्ज नहीं है।",
  "No screenings were recorded for the selected date range and age filter.":
    "चुनी गई तारीखों और आयु फ़िल्टर के लिए कोई जाँच दर्ज नहीं है।",
  "No matching actions, pages, or entities found.":
    "कोई मेल खाती कार्रवाई, पृष्ठ या जानकारी नहीं मिली।",
  "Type a command, page name, village, or worker...":
    "कमांड, पृष्ठ, गाँव या कार्यकर्ता का नाम लिखें...",
  "Overview Dashboard": "अवलोकन डैशबोर्ड",
  "Longitudinal Trends": "समय के साथ रुझान",
  "Location Risk Map": "स्थान जोखिम मानचित्र",
  "Geospatial Risk Map": "भौगोलिक जोखिम मानचित्र",
  "Reports & Exports": "रिपोर्ट और निर्यात",
  Reports: "रिपोर्ट",
  Locations: "स्थान",
  Workers: "कार्यकर्ता",
  Monitor: "निगरानी",
  Operate: "संचालन",
  Report: "रिपोर्ट",
  Updated: "अपडेट किया गया",
  "Refresh trends": "रुझान रीफ़्रेश करें",
  "Time-series screening data table": "समय के अनुसार जाँच डेटा तालिका",
  "Screening trends over time": "समय के साथ जाँच के रुझान",
  "Screening Volume & Risk Band Distribution": "जाँच संख्या और जोखिम श्रेणी का वितरण",
  "30-Day Screening Trend Overview": "30 दिनों के जाँच रुझान का अवलोकन",
  "Share of Programme": "कार्यक्रम का अनुपात",
  "Daily Average": "दैनिक औसत",
  "Period Total": "अवधि का कुल",
  "Total Scans": "कुल जाँच",
  "Total Scans:": "कुल जाँच:",
  "Total Screenings": "कुल जाँच",
  Total: "कुल",
  "Screened today": "आज जाँचे गए",
  "7 Pending": "7 लंबित",
  "14 Villages": "14 गाँव",
  Showing: "दिखाए जा रहे हैं",
  "Showing {count} locations": "{count} स्थान दिखाए जा रहे हैं",
  "Filter: All records (No age filter)": "फ़िल्टर: सभी रिकॉर्ड (आयु फ़िल्टर नहीं)",
  "Field team directory · Activity surveillance & performance tracking":
    "फील्ड टीम सूची · गतिविधि निगरानी और कार्य प्रदर्शन",
  "Geographic surveillance · Village risk mapping & field worker coverage":
    "भौगोलिक निगरानी · गाँव का जोखिम मानचित्र और कार्यकर्ता कवरेज",
  "Clinical triage workbench · Review SAM cases, low-confidence screenings, and child records":
    "चिकित्सकीय प्राथमिकता कार्यक्षेत्र · SAM मामले, कम विश्वास वाली जाँच और बच्चों के रिकॉर्ड देखें",
  "Export district health datasets for offline reporting or generate formal programme summaries.":
    "ऑफ़लाइन रिपोर्टिंग के लिए ज़िला स्वास्थ्य डेटा निर्यात करें या औपचारिक कार्यक्रम सारांश बनाएँ।",
  "Time-series workbench · longitudinal screening analytics across child screening cohorts":
    "समय-श्रृंखला कार्यक्षेत्र · बच्चों की जाँच के समूहों का दीर्घकालिक विश्लेषण",
  "Loading programme oversight data…": "कार्यक्रम निगरानी का डेटा लोड हो रहा है…",
  "Reports & Export": "रिपोर्ट और निर्यात",
  "District Health Report (CSV)": "ज़िला स्वास्थ्य रिपोर्ट (CSV)",
  "Download District CSV Report": "ज़िला CSV रिपोर्ट डाउनलोड करें",
  "Download CSV Dataset": "CSV डेटा डाउनलोड करें",
  "Generating CSV…": "CSV बनाया जा रहा है…",
  "Generated by the server for the selected age category.":
    "चुनी गई आयु श्रेणी के लिए सर्वर ने यह रिपोर्ट बनाई है।",
  "Retry Download": "डाउनलोड फिर से आज़माएँ",
  "Print / Save as PDF": "प्रिंट करें / PDF के रूप में सहेजें",
  "Exported CSV Preview Table": "निर्यात की गई CSV तालिका का पूर्वावलोकन",
  "Download the complete raw screening dataset for epidemiological analysis, HMIS reporting, and offline audits.":
    "महामारी विज्ञान विश्लेषण, HMIS रिपोर्टिंग और ऑफ़लाइन जाँच के लिए पूरा मूल जाँच डेटा डाउनलोड करें।",
  "Village Site": "गाँव केंद्र",
  Village: "गाँव",
  "SAM Cases": "SAM मामले",
  "MAM Cases": "MAM मामले",
  "SAM Rate": "SAM दर",
  "SAM Cases in Period": "अवधि में SAM मामले",
  "Average Confidence": "औसत विश्वास स्तर",
  "Avg Model Confidence": "मॉडल का औसत विश्वास स्तर",
  "Model confidence score of": "मॉडल का विश्वास स्कोर",
  "Quality / Sharpness": "गुणवत्ता / स्पष्टता",
  "Children & Flagged Scans": "बच्चे और चिह्नित जाँच",
  "Worker login": "कार्यकर्ता लॉग इन",
  Workflows: "कार्यप्रवाह",
  "Connected Operations": "जुड़ा हुआ संचालन",
  "Sort:": "क्रम:",
  "Refresh All Queries & Cache": "सभी क्वेरी और कैश रीफ़्रेश करें",
  "Return to PoshanScan Home": "PoshanScan होम पर लौटें",
  "API Inspector (Dev)": "API निरीक्षक (डेवलपर)",
} satisfies Record<string, string>;

const landing = {
  "Digital Screening for Frontline Child Nutrition":
    "बच्चों के पोषण के लिए अग्रिम पंक्ति की डिजिटल जाँच",
  "Digital Child Malnutrition Surveillance & Screening Summary":
    "बाल कुपोषण निगरानी और जाँच का डिजिटल सारांश",
  "National Child Nutrition Screening Initiative": "राष्ट्रीय बाल पोषण जाँच पहल",
  "Privacy-first child triage": "गोपनीयता को प्राथमिकता देने वाली बाल जाँच",
  "Two Roles, One Connected Workflow": "दो भूमिकाएँ, एक जुड़ा हुआ कार्यप्रवाह",
  "Frontline Health Worker Workflow": "अग्रिम पंक्ति के स्वास्थ्यकर्मी का कार्यप्रवाह",
  "Public Health Supervisor Workflow": "सार्वजनिक स्वास्थ्य पर्यवेक्षक का कार्यप्रवाह",
  "PoshanScan Platform": "PoshanScan मंच",
  "Programme Overview": "कार्यक्रम का अवलोकन",
  "Programme Summary Report": "कार्यक्रम सारांश रिपोर्ट",
  "Programme Summary (Print / PDF)": "कार्यक्रम सारांश (प्रिंट / PDF)",
  "Programme oversight,": "कार्यक्रम की निगरानी,",
  "Project & Research Team": "परियोजना और शोध टीम",
  "Ethical Safeguards": "नैतिक सुरक्षा उपाय",
  "Responsible Use": "ज़िम्मेदार उपयोग",
  "Frequently Asked Questions": "अक्सर पूछे जाने वाले सवाल",
  "Common Questions": "आम सवाल",
  "Technical, clinical, and operational answers about PoshanScan.":
    "PoshanScan के तकनीकी, चिकित्सकीय और संचालन संबंधी जवाब।",
  "How It Works": "यह कैसे काम करता है",
  "Scientific Pipeline": "वैज्ञानिक प्रक्रिया",
  "Geometric Ellipse Model": "ज्यामितीय दीर्घवृत्त मॉडल",
  "Limb Mask Segmentation": "बाँह के हिस्से का मास्क विभाजन",
  "MediaPipe Pose": "MediaPipe मुद्रा अनुमान",
  "OpenCV ArUco": "OpenCV ArUco",
  "YOLOv8 Segmentation": "YOLOv8 विभाजन",
  "Standard Camera Pose": "कैमरे की मानक स्थिति",
  "Midpoint Localization": "मध्य-बिंदु निर्धारण",
  "Optical scale: 0.18 mm/px": "ऑप्टिकल पैमाना: 0.18 mm/px",
  "Pose: Marker facing camera": "मुद्रा: मार्कर कैमरे की ओर",
  "Detects 4 outer corners with sub-pixel refinement to establish scale mm/px.":
    "mm/px पैमाना तय करने के लिए चार बाहरी कोनों का सटीक पता लगाता है।",
  "Automated joint keypoints isolate the midpoint between acromion and olecranon.":
    "जोड़ों के स्वचालित बिंदुओं से कंधे और कोहनी के बीच का मध्य-बिंदु पता चलता है।",
  "Removes background clothing, clutter, and shadows to isolate clean limb borders.":
    "बाँह की सीमा स्पष्ट करने के लिए पृष्ठभूमि के कपड़े, अव्यवस्था और छाया हटाता है।",
  "MUAC ≈ π · k · width. Evaluated against Shakir tape measurements (k ≈ 1.22–1.35).":
    "MUAC ≈ π · k · चौड़ाई। शाकिर टेप माप से इसका मूल्यांकन किया गया (k ≈ 1.22–1.35)।",
  "Applies to children aged 6 to 59 months. MUAC provides a rapid assessment of wasting without requiring height measurement in the field.":
    "यह 6 से 59 माह के बच्चों के लिए है। MUAC से फील्ड में लंबाई मापे बिना कुपोषण की त्वरित जाँच की जा सकती है।",
  "Drag the slider to test any mid-upper arm circumference from 80 mm to 180 mm. Observe how the clinical risk band and frontline referral guidance adapt in real time.":
    "80 mm से 180 mm तक बाँह की परिधि देखने के लिए स्लाइडर खिसकाएँ। देखें कि जोखिम श्रेणी और रेफ़रल सुझाव कैसे बदलते हैं।",
  "Built With Open Standard Technologies": "खुले मानकों वाली तकनीकों से निर्मित",
  "React 19 & TanStack Query": "React 19 और TanStack Query",
  "FastAPI & PostgreSQL": "FastAPI और PostgreSQL",
  "From Raw Smartphone Photo to Clinical Triage": "स्मार्टफ़ोन तस्वीर से चिकित्सकीय प्राथमिकता तक",
  "Health technologies must earn trust at the community level. PoshanScan is engineered around strict data minimization, client-side masking, and human-in-the-loop clinical protocols.":
    "स्वास्थ्य तकनीक को समुदाय का भरोसा जीतना चाहिए। PoshanScan कम से कम डेटा उपयोग, डिवाइस पर मास्किंग और स्वास्थ्यकर्मी की समीक्षा को प्राथमिकता देता है।",
  "PoshanScan estimates Mid-Upper Arm Circumference (MUAC) from a guided smartphone photograph next to a standard reference marker. Designed to empower community health workers with consistent, objective screening while automatically synchronizing flagged cases to public health supervisors.":
    "PoshanScan मानक संदर्भ मार्कर के साथ ली गई निर्देशित स्मार्टफ़ोन तस्वीर से मध्य-ऊपरी बाँह की परिधि (MUAC) का अनुमान लगाता है। यह समुदाय के स्वास्थ्यकर्मियों को एकसमान जाँच में मदद करता है और चिह्नित मामलों को पर्यवेक्षकों के साथ सिंक करता है।",
  "Frontline workers capture objective screenings on mobile devices in seconds. Supervisors review aggregated trends, verify flagged scans, and direct clinical follow-ups.":
    "अग्रिम पंक्ति के स्वास्थ्यकर्मी मोबाइल पर कुछ ही सेकंड में जाँच दर्ज करते हैं। पर्यवेक्षक कुल रुझान देखते हैं, चिह्नित जाँच की पुष्टि करते हैं और आगे की देखभाल का निर्देश देते हैं।",
  "PoshanScan turns a frontline smartphone camera into a calibrated screening instrument through an automated six-stage computer vision and geometric estimation pipeline.":
    "PoshanScan कंप्यूटर विज़न और ज्यामितीय अनुमान की छह चरणों वाली प्रक्रिया से स्मार्टफ़ोन कैमरे को माप-संतुलित जाँच उपकरण में बदलता है।",
  "Privacy, Governance & Responsible Deployment": "गोपनीयता, संचालन और ज़िम्मेदार उपयोग",
  "Illustrative example": "उदाहरण",
  "Interactive Reference": "इंटरैक्टिव संदर्भ",
  FAQ: "अक्सर पूछे जाने वाले सवाल",
  Documentation: "दस्तावेज़",
  "PoshanScan Initiative. All rights reserved.": "PoshanScan पहल। सर्वाधिकार सुरक्षित।",
  "Ready to Make Every Child Screening Count?":
    "हर बच्चे की जाँच को सार्थक बनाने के लिए तैयार हैं?",
  "Situation Room (Overview)": "स्थिति कक्ष (अवलोकन)",
  "Geospatial Risk Map": "भौगोलिक जोखिम मानचित्र",
  "Flagged Review Queue": "चिह्नित जाँच की समीक्षा सूची",
  "Low Conf": "कम विश्वास",
  "14 Villages": "14 गाँव",
  "7 Pending": "7 लंबित",
} satisfies Record<string, string>;

const additionalWorker = {
  "Add a child before capturing a screening.": "जाँच की तस्वीर लेने से पहले बच्चे को जोड़ें।",
  "Add an adult before capturing a screening.": "जाँच की तस्वीर लेने से पहले वयस्क को जोड़ें।",
  "Analyzing photo — this can take up to a minute":
    "तस्वीर का विश्लेषण हो रहा है — इसमें एक मिनट तक लग सकता है।",
  "Applied category": "लागू श्रेणी",
  "Auto-Sync": "स्वचालित सिंक",
  "Capture photo — PoshanScan": "तस्वीर लें — PoshanScan",
  Captured: "तस्वीर ली गई",
  "Captured At (IST)": "तस्वीर का समय (IST)",
  "Capture screening photo": "जाँच की तस्वीर लें",
  "Captured screening photo": "जाँच की ली गई तस्वीर",
  "Child MUAC scale (6–59 months)": "बच्चों का MUAC पैमाना (6–59 माह)",
  "Children 6–59 months": "6–59 माह के बच्चे",
  "Children — PoshanScan": "बच्चे — PoshanScan",
  "Child Nutrition Screening": "बाल पोषण जाँच",
  "Child is registered with normal surveillance status. No arm photo screening has been submitted yet.":
    "बच्चा सामान्य निगरानी स्थिति में पंजीकृत है। अभी बाँह की तस्वीर वाली कोई जाँच जमा नहीं हुई है।",
  "Date of birth": "जन्म तिथि",
  "Find a child": "बच्चा खोजें",
  "Find an adult": "वयस्क खोजें",
  "Find an adult to start a new screening.": "नई जाँच शुरू करने के लिए वयस्क खोजें।",
  "Find an existing adult or register a new adult before taking a photo.":
    "तस्वीर लेने से पहले मौजूदा वयस्क को खोजें या नया पंजीकरण करें।",
  "Find an existing child or register a new one before taking a photo.":
    "तस्वीर लेने से पहले मौजूदा बच्चे को खोजें या नया पंजीकरण करें।",
  "Find or register a child for nutrition screening.":
    "पोषण जाँच के लिए बच्चे को खोजें या पंजीकृत करें।",
  "Field Screening Site": "फील्ड जाँच केंद्र",
  "Find a child to start a new screening.": "नई जाँच शुरू करने के लिए बच्चा खोजें।",
  "Guided Photo": "निर्देशित तस्वीर",
  "Guided photo capture for MUAC screening.": "MUAC जाँच के लिए निर्देशित तस्वीर लें।",
  "Informed Caregiver Consent": "अभिभावक की जानकारी सहित सहमति",
  "Reference marker": "संदर्भ मार्कर",
  "Register a child": "बच्चे का पंजीकरण करें",
  "Register an adult": "वयस्क का पंजीकरण करें",
  "Use this photo": "यह तस्वीर इस्तेमाल करें",
  "Start a guided child nutrition screening.": "बच्चे की पोषण जाँच निर्देशों के अनुसार शुरू करें।",
  "Sign in to the PoshanScan field worker workspace.":
    "PoshanScan फील्ड कार्यकर्ता कार्यक्षेत्र में साइन इन करें।",
  "Sign in to the PoshanScan supervisor workspace.":
    "PoshanScan पर्यवेक्षक कार्यक्षेत्र में साइन इन करें।",
  "Log in": "लॉग इन करें",
} satisfies Record<string, string>;

const additionalScreening = {
  "Adult MUAC Screening Bands": "वयस्क MUAC जाँच श्रेणियाँ",
  "Adult MUAC scale": "वयस्क MUAC पैमाना",
  "Adequate Nutrition (Normal Range)": "पर्याप्त पोषण (सामान्य सीमा)",
  "Continue routine community monitoring & nutritional guidance":
    "समुदाय में नियमित निगरानी और पोषण मार्गदर्शन जारी रखें",
  "Continue routine community nutritional surveillance and follow-up screening according to schedule.":
    "निर्धारित समय के अनुसार समुदाय में नियमित पोषण निगरानी और आगे की जाँच जारी रखें।",
  "Confirm this result with a trained health worker.":
    "इस परिणाम की पुष्टि प्रशिक्षित स्वास्थ्यकर्मी से करें।",
  "Detailed MUAC screening report and recommended next steps.":
    "MUAC जाँच की विस्तृत रिपोर्ट और अगले सुझाए गए कदम।",
  "Evaluated against Shakir tape": "शाकिर टेप से तुलना की गई",
  "Moderate Acute Malnutrition": "मध्यम तीव्र कुपोषण",
  "Severe Acute Malnutrition": "गंभीर तीव्र कुपोषण",
  "Refer the adult promptly for clinical nutrition assessment according to your local protocol. Screening aid only — never a standalone diagnosis.":
    "स्थानीय प्रक्रिया के अनुसार वयस्क को शीघ्र पोषण संबंधी चिकित्सकीय जाँच के लिए भेजें। यह केवल जाँच में सहायता है, अपने आप में निदान नहीं।",
  "Refer the child immediately to a health facility for clinical assessment according to your local protocol. Screening aid only — never a standalone diagnosis.":
    "स्थानीय प्रक्रिया के अनुसार बच्चे को तुरंत चिकित्सकीय जाँच के लिए स्वास्थ्य केंद्र भेजें। यह केवल जाँच में सहायता है, अपने आप में निदान नहीं।",
  "Ask the field worker to repeat the adult measurement with a standard MUAC tape and verify camera angle, arm relaxation, and marker visibility.":
    "फील्ड कार्यकर्ता से मानक MUAC टेप द्वारा वयस्क का माप दोहराने को कहें। कैमरे का कोण, बाँह का ढीलापन और मार्कर की दृश्यता भी जाँचें।",
  "Ask the field worker to repeat the child measurement with a standard Shakir tape and verify camera angle, arm relaxation, and marker visibility.":
    "फील्ड कार्यकर्ता से मानक शाकिर टेप द्वारा बच्चे का माप दोहराने को कहें। कैमरे का कोण, बाँह का ढीलापन और मार्कर की दृश्यता भी जाँचें।",
  "Supplementary feeding counselling & close monitoring (re-screen in 2–4 wks)":
    "अतिरिक्त आहार पर सलाह और करीबी निगरानी (2–4 सप्ताह में फिर जाँचें)",
  "Urgent referral to Nutrition Rehabilitation Centre (NRC) for clinical assessment":
    "चिकित्सकीय जाँच के लिए पोषण पुनर्वास केंद्र (NRC) में तुरंत भेजें",
  "WHO Child Malnutrition Screening Bands (6–59 months)":
    "WHO बाल कुपोषण जाँच श्रेणियाँ (6–59 माह)",
  "WHO Alert Threshold (10–14.9%)": "WHO चेतावनी सीमा (10–14.9%)",
  "WHO Emergency Threshold (>15%)": "WHO आपात सीमा (>15%)",
  "Within expected limits (<10%)": "अपेक्षित सीमा के भीतर (<10%)",
  "Thresholds: SAM <": "सीमाएँ: SAM <",
  "WHO Guideline:": "WHO दिशानिर्देश:",
  "WHO screening criteria info": "WHO जाँच मानदंड की जानकारी",
  "Applied category": "लागू श्रेणी",
  "Screening Status": "जाँच की स्थिति",
  "Screening bands (6–59 months)": "जाँच श्रेणियाँ (6–59 माह)",
  "Screening bands (all categories)": "जाँच श्रेणियाँ (सभी वर्ग)",
  "Why This Scan Was Flagged": "यह जाँच क्यों चिह्नित हुई",
  "No reviewed scans yet": "अभी कोई जाँची हुई स्कैन नहीं है",
  "Reviewed scans will appear here after evaluation.":
    "मूल्यांकन के बाद जाँची हुई स्कैन यहाँ दिखाई देंगी।",
  "Queue clear — nothing needs review": "सूची खाली है — समीक्षा के लिए कुछ नहीं है",
  "All flagged cases have been successfully assessed and addressed.":
    "सभी चिह्नित मामलों की सफलतापूर्वक जाँच और कार्रवाई हो चुकी है।",
} satisfies Record<string, string>;

const additionalSupervisor = {
  "All Age Groups": "सभी आयु वर्ग",
  "All Children": "सभी बच्चे",
  "Flagged (pending)": "चिह्नित (लंबित)",
  "Flagged (reviewed)": "चिह्नित (समीक्षा पूरी)",
  "Flagged Only": "केवल चिह्नित",
  "Flagged Queue": "चिह्नित जाँच सूची",
  "Field Team Directory": "फील्ड टीम सूची",
  "Field workers directory": "फील्ड कार्यकर्ता सूची",
  "Field worker": "फील्ड कार्यकर्ता",
  "Field worker?": "फील्ड कार्यकर्ता?",
  "Worker ID": "कार्यकर्ता आईडी",
  "Worker area": "कार्य क्षेत्र",
  "Worker email": "कार्यकर्ता का ईमेल",
  "Worker phone": "कार्यकर्ता का फ़ोन",
  Gender: "लिंग",
  Guardian: "अभिभावक",
  "GPS coordinates": "GPS निर्देशांक",
  District: "ज़िला",
  "District Situation": "ज़िले की स्थिति",
  "Risk Mapping": "जोखिम मानचित्रण",
  Heatmap: "हीटमैप",
  Bubbles: "बुलबुले",
  Block: "ब्लॉक",
  Cohort: "समूह",
  Volume: "संख्या",
  "Volume Scale": "संख्या का पैमाना",
  "Chart View": "चार्ट दृश्य",
  "Table View": "तालिका दृश्य",
  Lines: "रेखाएँ",
  Stacked: "स्टैक्ड",
  Counts: "गिनती",
  "Share %": "अनुपात %",
  Day: "दिन",
  Week: "सप्ताह",
  "Interval:": "अंतराल:",
  "Reset Zoom": "ज़ूम रीसेट करें",
  "Reset brush zoom window": "ब्रश ज़ूम विंडो रीसेट करें",
  "Reset map view to encompass all coordinates":
    "सभी निर्देशांक दिखाने के लिए मानचित्र दृश्य रीसेट करें",
  "Copy Referrals": "रेफ़रल कॉपी करें",
  "Keyboard J/K review shortcuts enabled": "J/K कुंजियों से समीक्षा शॉर्टकट चालू हैं",
  "Live screening feed from all field workers": "सभी फील्ड कार्यकर्ताओं की लाइव जाँच जानकारी",
  "Busiest day": "सबसे व्यस्त दिन",
  "Period SAM share": "अवधि में SAM का अनुपात",
  "Period screenings": "अवधि की जाँच",
  "Percentage share of total screening volume over time (100% stacked)":
    "समय के साथ कुल जाँच का प्रतिशत (100% स्टैक्ड)",
  "Total screenings partitioned by WHO screening classification":
    "WHO जाँच वर्गीकरण के अनुसार कुल जाँच",
  "Trend analysis across districts and time": "ज़िलों और समय के अनुसार रुझान विश्लेषण",
  "Unknown site": "अज्ञात स्थान",
  Unassigned: "निर्धारित नहीं",
  "No children records found": "बच्चों के रिकॉर्ड नहीं मिले",
  "Try adjusting search query or filters.": "खोज या फ़िल्टर बदलकर देखें।",
  "Search records and begin screening.": "रिकॉर्ड खोजें और जाँच शुरू करें।",
  "Search records and begin an adult screening.": "रिकॉर्ड खोजें और वयस्क की जाँच शुरू करें।",
  "Search reference, child, village, worker...": "संदर्भ, बच्चा, गाँव या कार्यकर्ता खोजें...",
  "Search worker, phone, village...": "कार्यकर्ता, फ़ोन या गाँव खोजें...",
  "Copy referral summary": "रेफ़रल सारांश कॉपी करें",
  "Export CSV": "CSV निर्यात करें",
  "Run all endpoints": "सभी endpoints चलाएँ",
  "Running…": "चल रहा है…",
  "Retry failed": "विफल प्रयास फिर करें",
  "Saving…": "सहेजा जा रहा है…",
  "Open report": "रिपोर्ट खोलें",
  "View report": "रिपोर्ट देखें",
  "Sign in to start field screenings or review your district’s aggregated nutrition outcomes on the supervisor portal.":
    "फील्ड जाँच शुरू करने या पर्यवेक्षक पोर्टल पर ज़िले के पोषण परिणाम देखने के लिए साइन इन करें।",
  "API Inspector — PoshanScan Supervisor Dev": "API निरीक्षक — PoshanScan पर्यवेक्षक विकास",
  "Flagged Scans — PoshanScan Supervisor": "चिह्नित जाँच — PoshanScan पर्यवेक्षक",
  "Locations — PoshanScan Supervisor": "स्थान — PoshanScan पर्यवेक्षक",
  "Overview — PoshanScan Supervisor": "अवलोकन — PoshanScan पर्यवेक्षक",
  "Reports — PoshanScan Supervisor": "रिपोर्ट — PoshanScan पर्यवेक्षक",
  "Trends — PoshanScan Supervisor": "रुझान — PoshanScan पर्यवेक्षक",
  "Workers — PoshanScan Supervisor": "कार्यकर्ता — PoshanScan पर्यवेक्षक",
  "Supervisor login — PoshanScan": "पर्यवेक्षक लॉग इन — PoshanScan",
  "Worker login — PoshanScan": "कार्यकर्ता लॉग इन — PoshanScan",
  "Screening report — PoshanScan": "जाँच रिपोर्ट — PoshanScan",
  "New screening — PoshanScan": "नई जाँच — PoshanScan",
  "Capture photo — PoshanScan": "तस्वीर लें — PoshanScan",
  "PoshanScan — Digital Screening for Child Malnutrition": "PoshanScan — बाल कुपोषण की डिजिटल जाँच",
  "Smartphone-based MUAC screening and supervisor analytics platform for community child health.":
    "समुदाय में बाल स्वास्थ्य के लिए स्मार्टफ़ोन आधारित MUAC जाँच और पर्यवेक्षक विश्लेषण मंच।",
  "AI-assisted Mid-Upper Arm Circumference (MUAC) screening for frontline health workers and public health supervisors. Fast, consistent, digitally connected triage.":
    "अग्रिम पंक्ति के स्वास्थ्यकर्मियों और सार्वजनिक स्वास्थ्य पर्यवेक्षकों के लिए AI-सहायित मध्य-ऊपरी बाँह परिधि (MUAC) जाँच। तेज़, एकसमान और डिजिटल रूप से जुड़ी प्राथमिकता जाँच।",
  "Strict Screening-Aid Guardrails": "जाँच सहायता के लिए स्पष्ट सुरक्षा सीमाएँ",
  "Role-Based Access Control": "भूमिका आधारित पहुँच नियंत्रण",
  "Data Minimization & Masking": "डेटा न्यूनतम रखना और मास्किंग",
  "Toggle Theme (": "थीम बदलें (",
  "Dark Mode": "डार्क मोड",
  "Light Mode": "लाइट मोड",
  "Not provided": "नहीं दिया गया",
  Name: "नाम",
  ID: "आईडी",
  "Scan ID": "जाँच आईडी",
  "Applied category": "लागू श्रेणी",
  Standard: "मानक",
  "Open & Align": "खोलें और संरेखित करें",
  "Affix Marker": "मार्कर लगाएँ",
  STAGE: "चरण",
  STEP: "कदम",
  ESC: "ESC",
  "ESC for shortcut sheet": "शॉर्टकट सूची के लिए ESC",
  "g o": "g o",
  "g t": "g t",
  "g l": "g l",
  "g w": "g w",
  "g f": "g f",
  "g r": "g r",
  "Parse error:": "पार्स त्रुटि:",
  "parse ✗": "पार्स ✗",
  "parsed ✓": "पार्स ✓",
  'Click "Run all endpoints" to inspect the API.': "API देखने के लिए “Run all endpoints” चुनें।",
} satisfies Record<string, string>;

const additionalMisc = {
  "14 days": "14 दिन",
  "30 days": "30 दिन",
  "90 days": "90 दिन",
  "7 days": "7 दिन",
  "7-day MA": "7-दिन का औसत",
  "7d MA": "7-दिन का औसत",
  "Daily average": "दैनिक औसत",
  Active: "सक्रिय",
  "Active (≤24h)": "सक्रिय (≤24 घंटे)",
  "A−": "A−",
  A: "A",
  "A+": "A+",
  "awaiting review": "समीक्षा लंबित",
  "children)": "बच्चे)",
  adults: "वयस्क",
  adult: "वयस्क",
  child: "बच्चा",
  children: "बच्चे",
  cases: "मामले",
  locations: "स्थान",
  workers: "कार्यकर्ता",
  screenings: "जाँच",
  scans: "जाँच",
  "cols)": "कॉलम)",
  "rows)": "पंक्तियाँ)",
  "rows ·": "पंक्तियाँ ·",
  "% conf": "% विश्वास",
  "% is below the 60% reliability threshold.": "% की विश्वसनीयता 60% की सीमा से कम है।",
  "Low n": "कम नमूना संख्या",
  "99+": "99+",
  ESC: "ESC",
  of: "में से",
  mm: "mm",
  "mm SAM threshold.": "mm SAM सीमा।",
  "mm falls below the": "mm इससे कम है",
  "mm · MAM": "mm · MAM",
  "mm · Normal ≥": "mm · सामान्य ≥",
  "mm ≤ MUAC <": "mm ≤ MUAC <",
  "mm)": "mm)",
  "n=": "n=",
  "w ≈": "w ≈",
  "Estimate:": "अनुमान:",
  "—": "—",
  "▲": "▲",
  "▼": "▼",
  "✓ Ready (": "✓ तैयार (",
  "90°": "90°",
  "Already screened ·": "पहले ही जाँचे गए ·",
  "(Field Worker)": "(फील्ड कार्यकर्ता)",
  "(Secured backend storage requires authorized token access)":
    "(सुरक्षित बैकएंड स्टोरेज के लिए अधिकृत टोकन आवश्यक है)",
  "/ day": "/ दिन",
  "CartoDB Positron Layer": "CartoDB Positron परत",
  attributes: "विशेषताएँ",
  website: "वेबसाइट",
  "found. Register an": "मिला। पंजीकरण करें:",
  "location lacks": "स्थान में कमी है",
  "locations lack": "स्थानों में कमी है",
  summary: "सारांश",
  "% GAM": "% GAM",
  "MAM Band": "MAM श्रेणी",
  "SAM Band": "SAM श्रेणी",
  "Normal Band": "सामान्य श्रेणी",
  "Normal (≥125 mm)": "सामान्य (≥125 mm)",
  "Adult MUAC scale": "वयस्क MUAC पैमाना",
  "MAM Cases": "MAM मामले",
  "180 mm": "180 mm",
  "80 mm": "80 mm",
  "Name (A-Z)": "नाम (A-Z)",
} satisfies Record<string, string>;

const remainingCopy = {
  "#PS-8842 · 121 mm": "#PS-8842 · 121 mm",
  "#PS-9021 · 108 mm": "#PS-9021 · 108 mm",
  "<60% Conf": "<60% विश्वास",
  "≥ 10% (Critical)": "≥ 10% (गंभीर)",
  "≥ 15% (Critical)": "≥ 15% (गंभीर)",
  "Avg Confidence": "औसत विश्वास स्तर",
  "Backend classification for the selected cohort": "चुने गए समूह का बैकएंड वर्गीकरण",
  Columns: "कॉलम",
  "Command Menu": "कमांड मेनू",
  "Creating account…": "खाता बनाया जा रहा है…",
  "Dataset Preview (First 10 of": "डेटासेट पूर्वावलोकन (पहले 10, कुल",
  "Generate an official, printable A4 health briefing with KPI counts, WHO band breakdown, 30-day trends, and high-risk village listings.":
    "KPI गिनती, WHO श्रेणी विवरण, 30-दिन के रुझान और उच्च जोखिम वाले गाँवों की सूची वाली आधिकारिक A4 स्वास्थ्य रिपोर्ट बनाएँ।",
  "Geographic risk mapping and flagged-case review":
    "भौगोलिक जोखिम मानचित्रण और चिह्नित मामलों की समीक्षा",
  "Global Acute Malnutrition (MAM + SAM cases:": "कुल तीव्र कुपोषण (MAM + SAM मामले:",
  "Instant Band": "तुरंत श्रेणी",
  "LIVE SCREENING": "लाइव जाँच",
  "Loading report…": "रिपोर्ट लोड हो रही है…",
  "Login as Frontline Worker": "अग्रिम पंक्ति के कार्यकर्ता के रूप में लॉग इन करें",
  "Login as Supervisor": "पर्यवेक्षक के रूप में लॉग इन करें",
  "Login as Worker": "कार्यकर्ता के रूप में लॉग इन करें",
  "MUAC <": "MUAC <",
  "MUAC ≥": "MUAC ≥",
  "Model confidence": "मॉडल का विश्वास स्तर",
  "Monitoring Protocol:": "निगरानी प्रक्रिया:",
  "Needs follow-up": "आगे की देखभाल आवश्यक",
  "Normal (≥": "सामान्य (≥",
  "OF 06 ·": "06 में से ·",
  "Pending Flagged Scans (": "लंबित चिह्नित जाँच (",
  PoshanScan: "PoshanScan",
  "PoshanScan Programme Summary Report": "PoshanScan कार्यक्रम सारांश रिपोर्ट",
  "PoshanScan frontline child malnutrition screening (6–59 months) with reference marker":
    "संदर्भ मार्कर के साथ PoshanScan अग्रिम पंक्ति बाल कुपोषण जाँच (6–59 माह)",
  "PoshanScan is a digital smartphone screening aid designed for community triage. It is not a standalone diagnostic instrument. All identified MAM and SAM cases must be clinically confirmed by trained health personnel via standardized Shakir tape measurement and pediatric clinical assessment prior to therapeutic admission.":
    "PoshanScan समुदाय में प्राथमिक जाँच के लिए स्मार्टफ़ोन आधारित सहायता है, स्वतंत्र निदान उपकरण नहीं। MAM और SAM के सभी मामलों की उपचार से पहले प्रशिक्षित स्वास्थ्यकर्मी द्वारा मानक शाकिर टेप और बाल चिकित्सकीय जाँच से पुष्टि ज़रूरी है।",
  Registered: "पंजीकृत",
  Reviewed: "समीक्षा पूरी",
  Scan: "जाँच",
  "Scans flagged for SAM classification or model confidence < 60%":
    "SAM वर्गीकरण या 60% से कम मॉडल विश्वास वाली जाँच चिह्नित हैं",
  "Scope:": "दायरा:",
  "Screening Metadata": "जाँच का मेटाडेटा",
  Screenings: "जाँच",
  "Severe Acute Malnutrition (": "गंभीर तीव्र कुपोषण (",
  "Something went wrong": "कुछ गड़बड़ हुई",
  "TOTAL SCANS": "कुल जाँच",
  "Use the Worker login": "कार्यकर्ता लॉग इन का उपयोग करें",
  ms: "मि.से.",
  "to get started.": "शुरू करने के लिए।",
  "· G:": "· अभिभावक:",
  "Log in as a worker": "कार्यकर्ता के रूप में लॉग इन करें",
  "Log in as a supervisor": "पर्यवेक्षक के रूप में लॉग इन करें",
  "Reset Map View": "मानचित्र दृश्य रीसेट करें",
  "Refresh trends": "रुझान रीफ़्रेश करें",
  "Toggle Theme (": "थीम बदलें (",
  "Your session has expired or the credentials are incorrect. Please log in again.":
    "आपका सत्र समाप्त हो गया है या लॉग इन जानकारी सही नहीं है। कृपया फिर से लॉग इन करें।",
  "The server is temporarily unavailable. Please wait a moment and try again.":
    "सर्वर अभी उपलब्ध नहीं है। कृपया थोड़ी देर बाद फिर से प्रयास करें।",
  "The backend is temporarily unavailable. Please wait a moment and try again.":
    "बैकएंड अभी उपलब्ध नहीं है। कृपया थोड़ी देर बाद फिर से प्रयास करें।",
  "Something went wrong. Please try again.": "कुछ गड़बड़ हुई। कृपया फिर से प्रयास करें।",
  "An unexpected error occurred": "एक अनपेक्षित त्रुटि हुई।",
  "Unknown error": "अज्ञात त्रुटि",
  "The estimated MUAC is outside the supported validity range. No risk category was assigned.":
    "अनुमानित MUAC समर्थित सीमा से बाहर है, इसलिए जोखिम श्रेणी निर्धारित नहीं की गई। पूरी ऊपरी बाँह और मार्कर के साथ फिर से तस्वीर लें और मानक MUAC टेप से माप की पुष्टि करें।",
  "Could not connect to the AI/CV estimation service.": "AI/CV मापन सेवा से संपर्क नहीं हो सका।",
  "No image data provided.": "तस्वीर का डेटा नहीं मिला।",
  "Could not decode the uploaded file as an image.":
    "अपलोड की गई फ़ाइल को तस्वीर के रूप में पढ़ा नहीं जा सका।",
  "Image uploads must be 15 MB or smaller.": "तस्वीर का आकार 15 MB या उससे कम होना चाहिए।",
  "Decoded images must contain 20 megapixels or fewer.":
    "तस्वीर में 20 मेगापिक्सेल या उससे कम होने चाहिए।",
  "The trained YOLO segmentation checkpoint is not loaded. Configure a trained model before using /segment.":
    "प्रशिक्षित YOLO विभाजन मॉडल लोड नहीं है। /segment का उपयोग करने से पहले मॉडल सेट करें।",
  "The segmentation model could not process this image.":
    "विभाजन मॉडल इस तस्वीर को संसाधित नहीं कर सका।",
  "The image did not produce a finite MUAC measurement. Retake the photo with the arm and reference marker clearly visible.":
    "तस्वीर से मान्य MUAC माप नहीं मिला। बाँह और संदर्भ मार्कर साफ़ दिखाते हुए फिर से तस्वीर लें।",
  "The image did not produce a positive MUAC measurement. Retake the photo with the arm and reference marker clearly visible.":
    "तस्वीर से धनात्मक MUAC माप नहीं मिला। बाँह और संदर्भ मार्कर साफ़ दिखाते हुए फिर से तस्वीर लें।",
  "Retake the photo and confirm the measurement with a standard MUAC tape.":
    "फिर से तस्वीर लें और मानक MUAC टेप से माप की पुष्टि करें।",
  "A scan for this person and age category already exists.":
    "इस व्यक्ति और आयु श्रेणी की जाँच पहले से मौजूद है।",
  "Child record not found.": "बच्चे का रिकॉर्ड नहीं मिला।",
  "Enter your full name.": "अपना पूरा नाम लिखें।",
  "Enter a valid email address.": "मान्य ईमेल पता लिखें।",
  "Enter your village or field area.": "अपने गाँव या कार्य क्षेत्र का नाम लिखें।",
  "Password must be at least 8 characters.": "पासवर्ड कम से कम 8 अक्षरों का होना चाहिए।",
  "Confirm your password.": "अपने पासवर्ड की पुष्टि करें।",
  "Passwords do not match.": "पासवर्ड मेल नहीं खाते।",
  "Phone number or admin ID is required": "फ़ोन नंबर या व्यवस्थापक आईडी आवश्यक है।",
  "Password is required": "पासवर्ड आवश्यक है।",
  "Incorrect ID or password.": "आईडी या पासवर्ड सही नहीं है।",
  "Can't reach the server. Check your connection and try again.":
    "सर्वर से संपर्क नहीं हो सका। अपना कनेक्शन जाँचें और फिर से प्रयास करें।",
  "An unexpected error occurred.": "एक अनपेक्षित त्रुटि हुई।",
  Never: "कभी नहीं",
  "Just now": "अभी-अभी",
  Yesterday: "कल",
  "No activity recorded": "कोई गतिविधि दर्ज नहीं है",
  "{count}m ago": "{count} मिनट पहले",
  "{count}h ago": "{count} घंटे पहले",
  "{count}d ago": "{count} दिन पहले",
  "{count}w ago": "{count} सप्ताह पहले",
  "{count}mo ago": "{count} महीने पहले",
  "Scan {reference} marked as reviewed": "जाँच {reference} की समीक्षा पूरी हुई",
  "Scan {reference} was already reviewed or updated":
    "जाँच {reference} की समीक्षा पहले ही हो चुकी है या रिकॉर्ड अपडेट किया गया है",
  "Failed to review scan: {message}": "जाँच की समीक्षा नहीं हो सकी: {message}",
  "Successfully marked {count} scan as reviewed.": "{count} जाँच की समीक्षा पूरी चिह्नित की गई।",
  "Successfully marked {count} scans as reviewed.": "{count} जाँचों की समीक्षा पूरी चिह्नित की गई।",
  "{succeeded} reviewed, {failed} failed.": "{succeeded} की समीक्षा हुई, {failed} विफल हुईं।",
  "Referral summary copied to clipboard": "रेफ़रल सारांश क्लिपबोर्ड पर कॉपी किया गया",
  "District report downloaded ({size})": "ज़िला रिपोर्ट डाउनलोड हुई ({size})",
  "Export failed: {message}": "निर्यात विफल हुआ: {message}",
  "A scan for this person and age category ({category}) already exists.":
    "इस व्यक्ति और आयु श्रेणी ({category}) की जाँच पहले से मौजूद है।",
  "We couldn't detect the marker. Use the PoshanScan printed ArUco marker (usually 50 mm), keep the full square flat, sharp, and unobstructed in the photo, then retake.":
    "मार्कर नहीं मिला। PoshanScan का छपा हुआ ArUco मार्कर (आमतौर पर 50 मिमी) इस्तेमाल करें। पूरा चौकोर मार्कर सपाट, साफ़ और बिना रुकावट फ़्रेम में रखकर फिर से तस्वीर लें।",
  "We couldn't clearly see the upper arm. Retake with the whole upper arm in frame, good lighting, and the marker visible.":
    "ऊपरी बाँह साफ़ दिखाई नहीं दी। पूरी ऊपरी बाँह और मार्कर को फ़्रेम में रखते हुए अच्छी रोशनी में फिर से तस्वीर लें।",
  "Search {category} by name…": "नाम से {category} खोजें…",
  "View as chart": "चार्ट के रूप में देखें",
  "View as table": "तालिका के रूप में देखें",
  "Hide password": "पासवर्ड छिपाएँ",
  "Show password": "पासवर्ड दिखाएँ",
  "Expand sidebar": "साइडबार फैलाएँ",
  "Collapse sidebar": "साइडबार समेटें",
  "Sign out": "साइन आउट करें",
  "More pages": "और पृष्ठ",
  breadcrumb: "ब्रेडक्रंब नेविगेशन",
  pagination: "पृष्ठ नेविगेशन",
  "Toggle Sidebar": "साइडबार बदलें",
  "API online": "API चालू है",
  "API slow": "API धीमा है",
  "API offline": "API ऑफ़लाइन है",
  "Moderate risk (MAM)": "मध्यम जोखिम (MAM)",
  "High risk (SAM)": "उच्च जोखिम (SAM)",
  "Adult details": "वयस्क का विवरण",
  "Child details": "बच्चे का विवरण",
  Adult: "वयस्क",
  Child: "बच्चा",
  "Screening photo for {name}": "{name} की जाँच की तस्वीर",
  "This screening does not indicate elevated risk on the child MUAC scale. Continue routine growth and nutrition monitoring under your local protocol. Encourage age-appropriate nutritious feeding and seek assessment if the child becomes unwell or concerns arise. Repeat screening at the interval recommended by your programme.":
    "बच्चों के MUAC पैमाने पर इस जाँच में बढ़ा हुआ जोखिम नहीं दिखता। स्थानीय प्रक्रिया के अनुसार वृद्धि और पोषण की नियमित निगरानी जारी रखें। उम्र के अनुसार पौष्टिक आहार दें। बच्चा बीमार हो या चिंता हो तो स्वास्थ्यकर्मी से जाँच कराएँ। कार्यक्रम द्वारा सुझाए गए समय पर फिर जाँचें।",
  "This result indicates moderate screening risk and needs follow-up. Arrange assessment by a trained health worker to review the child's nutrition, health, and feeding. Provide counselling and any support recommended by your local programme. Schedule follow-up and repeat measurement according to the care protocol.":
    "यह परिणाम मध्यम जोखिम दिखाता है और आगे की जाँच आवश्यक है। बच्चे के पोषण, स्वास्थ्य और आहार की समीक्षा के लिए प्रशिक्षित स्वास्थ्यकर्मी से मिलें। स्थानीय कार्यक्रम के अनुसार परामर्श और सहायता लें। देखभाल प्रक्रिया के अनुसार आगे की मुलाक़ात और दोबारा माप तय करें।",
  "This result indicates high screening risk and requires prompt clinical assessment. Refer the child urgently to an appropriate health facility or nutrition service under your local protocol. A trained clinician should assess for complications and determine the right treatment setting. If the child is very unwell or has danger signs, seek emergency care immediately.":
    "यह परिणाम अधिक जोखिम दिखाता है और शीघ्र चिकित्सकीय जाँच आवश्यक है। स्थानीय प्रक्रिया के अनुसार बच्चे को तुरंत उचित स्वास्थ्य केंद्र या पोषण सेवा में भेजें। प्रशिक्षित चिकित्सक जटिलताओं की जाँच करके उचित उपचार तय करें। बच्चा बहुत बीमार हो या खतरे के संकेत हों तो तुरंत आपात सहायता लें।",
  "This screening does not indicate elevated risk on the adult MUAC scale. Continue routine nutrition and health monitoring under your local protocol. Encourage a varied, adequate diet where possible. Repeat screening at the interval recommended by your programme or sooner if concerns arise.":
    "वयस्क MUAC पैमाने पर इस जाँच में बढ़ा हुआ जोखिम नहीं दिखता। स्थानीय प्रक्रिया के अनुसार पोषण और स्वास्थ्य की नियमित निगरानी जारी रखें। जहाँ संभव हो, विविध और पर्याप्त आहार लें। कार्यक्रम द्वारा सुझाए गए समय पर या चिंता होने पर पहले फिर जाँचें।",
  "This result indicates moderate screening risk and should be followed up. Arrange assessment by a trained health worker to review nutrition status and possible contributing health or dietary factors. Follow local guidance for counselling, support, and any further assessment. Agree on a follow-up plan and repeat measurement as recommended.":
    "यह परिणाम मध्यम जोखिम दिखाता है और आगे की देखभाल आवश्यक है। पोषण स्थिति और स्वास्थ्य या आहार से जुड़े संभावित कारणों की समीक्षा के लिए प्रशिक्षित स्वास्थ्यकर्मी से मिलें। परामर्श, सहायता और आगे की जाँच के लिए स्थानीय मार्गदर्शन लें। आगे की देखभाल की योजना बनाकर सुझाए अनुसार दोबारा मापें।",
  "This result indicates high screening risk and needs prompt clinical assessment. Refer the adult to an appropriate health facility or nutrition service under your local protocol. A trained clinician should assess overall health and determine the appropriate care. If the person is acutely unwell, seek emergency care immediately.":
    "यह परिणाम अधिक जोखिम दिखाता है और शीघ्र चिकित्सकीय जाँच आवश्यक है। स्थानीय प्रक्रिया के अनुसार वयस्क को उचित स्वास्थ्य केंद्र या पोषण सेवा में भेजें। प्रशिक्षित चिकित्सक स्वास्थ्य की जाँच करके उचित देखभाल तय करें। व्यक्ति अचानक बहुत बीमार हो तो तुरंत आपात सहायता लें।",
} satisfies Record<string, string>;

const applicationUi = {
  Sidebar: "साइडबार",
  Navigation: "नेविगेशन",
  "Cohort & Filters": "समूह और फ़िल्टर",
  "Quick Actions": "त्वरित कार्रवाइयाँ",
  "Jump to Entity": "रिकॉर्ड पर जाएँ",
  "Response format unexpected from supervisor overview endpoint.":
    "पर्यवेक्षक अवलोकन एंडपॉइंट से प्राप्त उत्तर का प्रारूप अपेक्षित नहीं था।",
  "Unable to retrieve programme overview metrics.":
    "कार्यक्रम के अवलोकन आँकड़े प्राप्त नहीं किए जा सके।",
  Worker: "कार्यकर्ता",
  "Assigned Villages": "निर्धारित गाँव",
  "Last Active": "अंतिम सक्रियता",
  Status: "स्थिति",
  Phone: "फ़ोन",
  "Avg Confidence %": "औसत विश्वास %",
  "Flagged Count": "चिह्नित जाँचों की संख्या",
  "Activity Status": "गतिविधि स्थिति",
  Idle: "निष्क्रिय",
  Dormant: "लंबे समय से निष्क्रिय",
  "Flagged ({count})": "चिह्नित ({count})",
  "Reviewing {current}/{total}...": "समीक्षा जारी है {current}/{total}...",
  "Mark {count} Reviewed": "{count} की समीक्षा पूरी करें",
  "Normal (≥{value} mm)": "सामान्य (≥{value} mm)",
  "MAM ({low}–<{high} mm)": "MAM ({low}–<{high} mm)",
  "SAM (<{value} mm)": "SAM (<{value} mm)",
  "{percentage}% of total": "कुल का {percentage}%",
  "{value} screenings/day avg": "औसतन {value} जाँच/दिन",
  "{count} screenings": "{count} जाँच",
  "Total scans": "कुल जाँच",
  "Model prediction score": "मॉडल पूर्वानुमान स्कोर",
  More: "और",
  carousel: "कैरोसेल",
  slide: "स्लाइड",
  "Previous slide": "पिछली स्लाइड",
  "Next slide": "अगली स्लाइड",
  "Go to previous page": "पिछले पृष्ठ पर जाएँ",
  "Go to next page": "अगले पृष्ठ पर जाएँ",
  Children: "बच्चे",
  "Search adults by name…": "नाम से वयस्क खोजें…",
  "Search children by name…": "नाम से बच्चे खोजें…",
  "Search villages...": "गाँव खोजें...",
  "Search workers...": "कार्यकर्ता खोजें...",
  "SAM Band: < {value} mm": "SAM श्रेणी: < {value} mm",
  "MAM Band: {low} – {high} mm": "MAM श्रेणी: {low} – {high} mm",
  "Normal Band: ≥ {value} mm": "सामान्य श्रेणी: ≥ {value} mm",
  "Normal: {count} ({pct}%)": "सामान्य: {count} ({pct}%)",
  "MAM: {count} ({pct}%)": "MAM: {count} ({pct}%)",
  "SAM: {count} ({pct}%)": "SAM: {count} ({pct}%)",
  "Displays the mobile sidebar.": "मोबाइल साइडबार दिखाता है।",
  "–<": "–<",
  "Project Lead & System Architect": "परियोजना प्रमुख और सिस्टम आर्किटेक्ट",
  "Lead Researcher": "प्रमुख शोधकर्ता",
  "PoshanScan Research Initiative": "PoshanScan अनुसंधान पहल",
  "Computer vision and field epidemiology integration.":
    "कंप्यूटर विज़न और क्षेत्रीय महामारी-विज्ञान का एकीकरण।",
  "Clinical Nutrition Advisor": "नैदानिक पोषण सलाहकार",
  "Public Health Consultant": "जन-स्वास्थ्य सलाहकार",
  "Child Nutrition & Triage Protocols": "बाल पोषण और प्राथमिक जाँच प्रोटोकॉल",
  "Guidance on WHO MUAC screening guidelines and referral workflows.":
    "WHO MUAC जाँच दिशानिर्देशों और रेफ़रल प्रक्रिया पर मार्गदर्शन।",
  "Frontend & Field UX Specialist": "फ्रंटएंड और फील्ड UX विशेषज्ञ",
  "Lead Software Engineer": "प्रमुख सॉफ़्टवेयर इंजीनियर",
  "Community Health Systems": "सामुदायिक स्वास्थ्य प्रणालियाँ",
  "Supervisor portal architecture, accessibility, and offline frontline workflows.":
    "पर्यवेक्षक पोर्टल की संरचना, सुगम्यता और ऑफ़लाइन फील्ड कार्यप्रवाह।",
  "Machine Learning Engineer": "मशीन लर्निंग इंजीनियर",
  "Computer Vision Researcher": "कंप्यूटर विज़न शोधकर्ता",
  "Model Calibration & Marker Detection": "मॉडल अंशांकन और मार्कर पहचान",
  "ArUco detection, arm segmentation, and geometric circumference estimation.":
    "ArUco पहचान, बाँह का विभाजन और ज्यामितीय परिधि अनुमान।",
  "Children 6–59 months · All villages": "6–59 माह के बच्चे · सभी गाँव",
  "Adults · All villages": "वयस्क · सभी गाँव",
  "All age records · All villages": "सभी आयु के रिकॉर्ड · सभी गाँव",
  worker: "कार्यकर्ता",
  supervisor: "पर्यवेक्षक",
  admin: "व्यवस्थापक",
  "Moderate Acute Malnutrition": "मध्यम तीव्र कुपोषण",
  "Adequate Nutrition (Normal)": "पर्याप्त पोषण (सामान्य)",
  "Severe Acute Malnutrition": "गंभीर तीव्र कुपोषण",
  "Standardized Capture": "मानकीकृत तस्वीर लेना",
  "Marker & arm alignment": "मार्कर और बाँह का सही संरेखण",
  "Frontline worker aligns camera perpendicular to the upper arm midpoint with printed reference marker affixed flat.":
    "फील्ड कार्यकर्ता मुद्रित संदर्भ मार्कर को बाँह पर सपाट लगाकर, कैमरे को बाँह के मध्य बिंदु के सामने सीधा रखता है।",
  "Guidance overlays check for camera distance, tilt, and stability before triggering capture.":
    "तस्वीर लेने से पहले निर्देश कैमरे की दूरी, झुकाव और स्थिरता की जाँच करते हैं।",
  "Marker Detection": "मार्कर पहचान",
  "Sub-pixel corner detection": "पिक्सेल से भी सूक्ष्म कोनों की पहचान",
  "ArUco algorithm locates the 4 marker corners to compute the exact optical scale in millimeters per pixel.":
    "ArUco एल्गोरिदम मार्कर के चारों कोनों की पहचान करके प्रति पिक्सेल मिलीमीटर में सटीक मापनी निकालता है।",
  "Compensates for perspective distortion and establishes a metric reference independent of smartphone sensor size.":
    "यह परिप्रेक्ष्य विकृति की भरपाई करता है और फ़ोन सेंसर के आकार से स्वतंत्र माप संदर्भ देता है।",
  "Arm Localization": "बाँह की स्थिति पहचान",
  "Anatomical landmarking": "शरीर के मुख्य बिंदुओं की पहचान",
  "Computer vision isolates the upper arm anatomy between the shoulder joint and elbow bend.":
    "कंप्यूटर विज़न कंधे के जोड़ और कोहनी के मोड़ के बीच के ऊपरी बाँह क्षेत्र की पहचान करता है।",
  "Identifies the standard WHO MUAC measurement level (halfway point) automatically.":
    "यह WHO के मानक MUAC माप स्तर यानी बीच के बिंदु को अपने आप पहचानता है।",
  "Contour Segmentation": "सीमा का विभाजन",
  "Sub-pixel edge boundary": "पिक्सेल से भी सूक्ष्म किनारे की सीमा",
  "Deep semantic segmentation separates arm skin boundary from background clothing, shadows, and environment.":
    "डीप सेमैंटिक विभाजन बाँह की बाहरी सीमा को कपड़ों, छाया और पृष्ठभूमि से अलग करता है।",
  "Robust edge localization extracts the true cross-sectional silhouette width at the midpoint level.":
    "किनारों की सटीक पहचान मध्य बिंदु पर बाँह की वास्तविक अनुप्रस्थ चौड़ाई निकालती है।",
  "Geometric Estimation": "ज्यामितीय अनुमान",
  "Calibrated ellipse model": "कैलिब्रेट किया हुआ दीर्घवृत्त मॉडल",
  "Physical diameter is scaled (px × mm/px) and converted into circumference via empirically calibrated cross-section factor k.":
    "भौतिक व्यास को मापनी के अनुसार बदला जाता है और अनुभवजन्य रूप से कैलिब्रेट किए गए अनुप्रस्थ कारक k से परिधि निकाली जाती है।",
  "The scaling factor k (1.22–1.35) accounts for upper arm cross-sectional ellipticity evaluated rigorously against Shakir tape.":
    "मापनी कारक k (1.22–1.35) ऊपरी बाँह के अंडाकार अनुप्रस्थ आकार को दर्शाता है और Shakir टेप के आधार पर जाँचा गया है।",
  "WHO Band Triage": "WHO श्रेणी के अनुसार प्राथमिक जाँच",
  "Clinical triage classification": "नैदानिक प्राथमिक जाँच वर्गीकरण",
  "Instant assignment into WHO child screening bands: Normal (≥125 mm), MAM (115–<125 mm), and SAM (<115 mm).":
    "बच्चों की जाँच को तुरंत WHO श्रेणियों में रखा जाता है: सामान्य (≥125 mm), MAM (115–<125 mm) और SAM (<115 mm)।",
  "Provides unambiguous frontline guidance while securely streaming flagged readings to the supervisor portal.":
    "यह फील्ड कार्यकर्ताओं को स्पष्ट निर्देश देता है और चिह्नित माप सुरक्षित रूप से पर्यवेक्षक पोर्टल पर भेजता है।",
  "What is Mid-Upper Arm Circumference (MUAC) and why is it used?":
    "मध्य ऊपरी बाँह की परिधि (MUAC) क्या है और इसका उपयोग क्यों किया जाता है?",
  "MUAC is a quick, reliable indicator of muscle and fat wasting in children aged 6–59 months. It is strongly correlated with child mortality risk and is the global standard for community-level acute malnutrition screening.":
    "MUAC, 6–59 माह के बच्चों में मांसपेशियों और वसा की कमी पहचानने का एक तेज़ संकेतक है। इसका संबंध बाल मृत्यु के जोखिम से है और समुदाय स्तर पर तीव्र कुपोषण की जाँच के लिए यह वैश्विक मानक है।",
  "How does PoshanScan calculate MUAC without a physical tape measure?":
    "PoshanScan बिना मापने वाले टेप के MUAC कैसे निकालता है?",
  "The user places a calibrated 50 mm ArUco marker on the arm. The computer vision pipeline detects the marker corners to obtain an exact scale factor (mm per pixel), segments the arm silhouette, and computes circumference using a validated geometric ellipse formula (MUAC ≈ π · k · width, where k is an empirically calibrated factor between 1.22 and 1.35).":
    "उपयोगकर्ता बाँह पर कैलिब्रेट किया हुआ 50 mm ArUco मार्कर लगाता है। कंप्यूटर विज़न प्रक्रिया मार्कर के कोनों से सटीक मापनी (mm प्रति पिक्सेल) निकालती है, बाँह की सीमा पहचानती है और ज्यामितीय दीर्घवृत्त सूत्र से परिधि निकालती है (MUAC ≈ π · k · चौड़ाई; k का अनुभवजन्य कैलिब्रेशन 1.22 से 1.35 के बीच है)।",
  "Does PoshanScan provide a medical diagnosis?": "क्या PoshanScan चिकित्सीय निदान देता है?",
  "No. PoshanScan is strictly a community triage and screening aid designed to assist frontline workers. Any child flagged as MAM or SAM must be clinically evaluated and confirmed with standard Shakir tape measurements by trained healthcare staff before medical interventions.":
    "नहीं। PoshanScan समुदाय स्तर की प्राथमिक जाँच में फील्ड कार्यकर्ताओं की सहायता का साधन है। MAM या SAM के रूप में चिह्नित हर बच्चे की प्रशिक्षित स्वास्थ्यकर्मी द्वारा चिकित्सीय जाँच और मानक Shakir टेप से पुष्टि आवश्यक है।",
  "Can frontline workers use PoshanScan in low-connectivity areas?":
    "क्या कम नेटवर्क वाले क्षेत्रों में फील्ड कार्यकर्ता PoshanScan इस्तेमाल कर सकते हैं?",
  "Yes. The frontline worker interface is built with an offline-first architecture. Scans and initial assessments are captured locally on the device and automatically synchronized to the supervisor dashboard once connectivity is restored.":
    "हाँ। फील्ड कार्यकर्ताओं का इंटरफ़ेस पहले ऑफ़लाइन काम करने के लिए बनाया गया है। जाँच और शुरुआती आकलन डिवाइस पर सहेजे जाते हैं और नेटवर्क लौटने पर पर्यवेक्षक डैशबोर्ड से अपने आप सिंक हो जाते हैं।",
  "How is child privacy protected?": "बच्चों की निजता कैसे सुरक्षित रखी जाती है?",
  "The platform adheres to strict data minimization. Child names are masked by default across all dashboards with a controlled reveal toggle. Only authorized supervisors have access to district triage logs, and no biometric identities are stored or sold.":
    "यह प्लेटफ़ॉर्म केवल आवश्यक डेटा रखता है। सभी डैशबोर्ड में बच्चों के नाम डिफ़ॉल्ट रूप से छिपे रहते हैं और अनुमति मिलने पर ही दिखते हैं। ज़िला जाँच रिकॉर्ड केवल अधिकृत पर्यवेक्षक देख सकते हैं; बायोमेट्रिक पहचान न तो सहेजी जाती है, न बेची जाती है।",
  "Open PoshanScan on mobile; review camera angle guidelines with zero setup friction.":
    "मोबाइल पर PoshanScan खोलें और बिना जटिल सेटअप के कैमरा कोण के निर्देश देखें।",
  "Place the 50 mm reference ArUco marker at child's mid-upper arm point.":
    "50 mm संदर्भ ArUco मार्कर को बच्चे की ऊपरी बाँह के मध्य बिंदु पर लगाएँ।",
  "Viewfinder confirms marker corners and distance before triggering rapid capture.":
    "तेज़ी से तस्वीर लेने से पहले व्यूफ़ाइंडर मार्कर के कोनों और दूरी की पुष्टि करता है।",
  "Device estimates MUAC and indicates WHO band (Normal, MAM, or SAM).":
    "डिवाइस MUAC का अनुमान लगाता है और WHO श्रेणी (सामान्य, MAM या SAM) दिखाता है।",
  "Screening record streams securely to the supervisor portal (offline queue supported).":
    "जाँच रिकॉर्ड सुरक्षित रूप से पर्यवेक्षक पोर्टल पर भेजा जाता है; ऑफ़लाइन कतार भी उपलब्ध है।",
  "Monitor live KPI cards: SAM prevalence, active workers, and volume across blocks.":
    "SAM के अनुपात, सक्रिय कार्यकर्ताओं और ब्लॉक-स्तर की जाँच संख्या वाले लाइव KPI कार्ड देखें।",
  "Geospatial cluster map pinpointing villages with highest severe wasting rates.":
    "भौगोलिक क्लस्टर मानचित्र में गंभीर कुपोषण की सबसे अधिक दर वाले गाँव देखें।",
  "Master-detail review queue isolating SAM and low-confidence scans for validation.":
    "SAM और कम विश्वास वाली जाँचों को सत्यापन के लिए अलग दिखाने वाली समीक्षा कतार देखें।",
  "Generate clinical referral summaries directly into clipboard for local health posts.":
    "स्थानीय स्वास्थ्य केंद्रों के लिए चिकित्सीय रेफ़रल सारांश सीधे क्लिपबोर्ड पर बनाएँ।",
  "One-click district CSV downloads with Bearer token authentication and print-ready A4 reports.":
    "एक क्लिक में ज़िला CSV डाउनलोड करें और प्रिंट के लिए तैयार A4 रिपोर्ट बनाएँ।",
  "Active ({count})": "सक्रिय ({count})",
  "Idle ({count})": "निष्क्रिय ({count})",
  "Dormant ({count})": "लंबे समय से निष्क्रिय ({count})",
} satisfies Record<string, string>;

const modules = [
  common,
  auth,
  worker,
  screening,
  supervisor,
  landing,
  additionalWorker,
  additionalScreening,
  additionalSupervisor,
  additionalMisc,
  remainingCopy,
  applicationUi,
];
const hindiCatalog = Object.assign({}, ...modules) as Record<string, string>;
const warnedMissing = new Set<string>();
const errorCodeCatalog: Record<string, string> = {
  invalid_image: "कृपया एक मान्य तस्वीर अपलोड करें।",
  invalid_measurement:
    "अनुमानित MUAC समर्थित सीमा से बाहर है, इसलिए जोखिम श्रेणी निर्धारित नहीं की गई। पूरी ऊपरी बाँह और मार्कर के साथ फिर से तस्वीर लें और मानक MUAC टेप से माप की पुष्टि करें।",
  cv_service_unavailable: "जाँच सेवा अभी उपलब्ध नहीं है। कृपया थोड़ी देर बाद फिर से प्रयास करें।",
  scan_already_exists: "इस व्यक्ति और आयु श्रेणी की जाँच पहले से मौजूद है।",
  image_too_large: "तस्वीर का आकार 15 MB या उससे कम होना चाहिए।",
  image_dimensions_too_large: "तस्वीर में 20 मेगापिक्सेल या उससे कम होने चाहिए।",
  segmentation_model_unavailable: "तस्वीर विभाजन सेवा अभी उपलब्ध नहीं है।",
  segmentation_failed:
    "ऊपरी बाँह का मॉडल साफ़ बाँह मास्क नहीं पहचान सका। पूरी ऊपरी बाँह को फ़्रेम में रखते हुए, मार्कर को सपाट और पूरा दिखाकर फिर से तस्वीर लें।",
  internal_error: "तस्वीर संसाधित करते समय अनपेक्षित त्रुटि हुई।",
  pipeline_failed:
    "जाँच पूरी नहीं हो सकी। बाँह और संदर्भ मार्कर साफ़ दिखाते हुए फिर से तस्वीर लें।",
};
const errorCodeEnglish: Record<string, string> = {
  invalid_image: "Please upload a valid image file.",
  invalid_measurement:
    "The estimated MUAC is outside the supported range, so no risk category was assigned. Retake with the full upper arm and marker visible, then confirm with a standard MUAC tape.",
  cv_service_unavailable:
    "The screening service is temporarily unavailable. Please try again shortly.",
  scan_already_exists: "A scan for this person and age category already exists.",
  image_too_large: "Image uploads must be 15 MB or smaller.",
  image_dimensions_too_large: "Decoded images must contain 20 megapixels or fewer.",
  segmentation_model_unavailable: "The segmentation service is temporarily unavailable.",
  segmentation_failed:
    "The upper-arm model could not identify a clear arm mask. Retake with the full upper arm visible, the marker flat and fully visible, and the arm separated from clothing and background.",
  internal_error: "An unexpected error occurred while processing the image.",
  pipeline_failed:
    "The screening could not be completed. Retake the photo with the arm and reference marker clearly visible.",
};

function storedLanguage(): PortalLanguage {
  try {
    const saved = JSON.parse(localStorage.getItem("poshanscan_portal_prefs") || "{}");
    return saved.language === "hi" ? "hi" : "en";
  } catch {
    return "en";
  }
}

export function normalizeTranslationKey(source: string): string {
  return source
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&ge;/g, "\u2265")
    .replace(/&le;/g, "\u2264")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function translate(source: string, language: PortalLanguage = storedLanguage()): string {
  const key = normalizeTranslationKey(source);
  if (import.meta.env.DEV && key && !hindiCatalog[key] && !warnedMissing.has(key)) {
    warnedMissing.add(key);
    console.warn(`[i18n] Missing Hindi translation: ${key}`);
  }
  if (language === "en") {
    const reverseEntry = Object.entries(hindiCatalog).find(([, hindi]) => hindi === key)?.[0];
    if (reverseEntry) return reverseEntry;
    const reverseErrorCode = Object.entries(errorCodeCatalog).find(
      ([, hindi]) => hindi === key,
    )?.[0];
    if (reverseErrorCode) return errorCodeEnglish[reverseErrorCode] ?? key;
    return key;
  }
  const translated = hindiCatalog[key];
  if (translated) return translated;
  if (Object.values(errorCodeCatalog).includes(key)) return key;
  return key;
}

export function t(source: string, values: Record<string, string | number> = {}): string {
  return Object.entries(values).reduce(
    (message, [name, value]) => message.replaceAll(`{${name}}`, String(value)),
    translate(source),
  );
}

export function translateApiMessage(message: string, code?: string): string {
  const language = storedLanguage();
  if (language === "hi" && code && errorCodeCatalog[code]) return errorCodeCatalog[code];
  if (language === "en" && code && errorCodeEnglish[code]) return errorCodeEnglish[code];
  return translate(message, language);
}

export function getCurrentLocale(): "en-IN" | "hi-IN" {
  return storedLanguage() === "hi" ? "hi-IN" : "en-IN";
}

export function plural(
  count: number,
  messages: { one: string; other: string },
  values: Record<string, string | number> = {},
): string {
  const language = storedLanguage();
  const category = new Intl.PluralRules(language === "hi" ? "hi-IN" : "en-IN").select(count);
  return t(category === "one" ? messages.one : messages.other, {
    ...values,
    count: new Intl.NumberFormat(language === "hi" ? "hi-IN" : "en-IN").format(count),
  });
}

export function formatNumber(value: number, options?: Intl.NumberFormatOptions): string {
  return new Intl.NumberFormat(getCurrentLocale(), options).format(value);
}

export function formatDate(
  value: Date | string | number,
  options?: Intl.DateTimeFormatOptions,
): string {
  return new Intl.DateTimeFormat(getCurrentLocale(), options).format(
    value instanceof Date ? value : new Date(value),
  );
}

export function T({ source, children }: { source?: string; children?: ReactNode }) {
  const { language } = usePortalPrefs();
  if (source !== undefined) return <>{translate(source, language)}</>;
  if (typeof children === "string") return <>{translate(children, language)}</>;
  return <>{children}</>;
}

export function translationKeys() {
  return Object.keys(hindiCatalog);
}

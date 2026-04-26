from deep_translator import GoogleTranslator

test_str1 = "we will give a firm reply to terrorists"
test_str2 = "we will give a firm reply to aatankwaadis"
test_str3 = "हम आतंकवादियों को कड़ा जवाब देंगे"

translator = GoogleTranslator(source='auto', target='en')
print("English:", translator.translate(test_str1))
print("Hinglish:", translator.translate(test_str2))
print("Hindi:", translator.translate(test_str3))

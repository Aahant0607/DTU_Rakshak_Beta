import base64
import streamlit as st
import string
import re

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# # Initialize the OCR reader
# reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

import re

def is_valid_license_format(text):
    """
    Validates standard Indian license plates ranging from 8 to 11 characters.
    Handles standard states (HR26A1234), Delhi formats (DL1CAA1234, DL10CAA1234),
    and older plates without series letters (MH121234).
    """
    text = ''.join(text.split())
    
    # [State: 2 letters] [RTO: 1 or 2 digits] [Series/Class: 0 to 3 letters] [Number: 4 digits]
    # Removed ^ and $ so it can search INSIDE a messy string
    general_format = r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}'
    
    # Changed from re.match to re.search
    return bool(re.search(general_format, text))

def format_license(text):
    """
    Format the license plate text by converting confusing characters (like O and 0).
    Dynamically handles 9 and 10 character plates.
    """
    license_plate_ = ''
    
    # Define mapping based on character position (Letters vs Numbers)
    if len(text) == 10:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int, 
                   4: dict_int_to_char, 5: dict_int_to_char, 6: dict_char_to_int, 7: dict_char_to_int, 
                   8: dict_char_to_int, 9: dict_char_to_int}
    elif len(text) == 9:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_int_to_char, 
                   4: dict_int_to_char, 5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int, 
                   8: dict_char_to_int}
    else:
        return text # Return raw if it doesn't match standard length
    
    for j in range(len(text)):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_
"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

def fix_general_label_error(slot, value):

    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"yes",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others 
        "y":"yes", "any":"do not care", "n":"no", "does not care":"do not care", "dontcare": "do not care" , "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
        }

    if value in GENERAL_TYPO.keys():
        value = GENERAL_TYPO[value]

    # miss match slot and value 
    if  slot == "hotel-type" and value in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
        slot == "hotel-internet" and value == "4" or \
        slot == "hotel-pricerange" and value == "2" or \
        slot == "attraction-type" and value in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
        "area" in slot and value in ["moderate"] or \
        "day" in slot and value == "t":
        value = "none"
    elif slot == "hotel-type" and value in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
        value = "hotel"
    elif slot == "hotel-stars" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no": value = "north"
        elif value == "we": value = "west"
        elif value == "cent": value = "centre"
    elif "day" in slot:
        if value == "we": value = "wednesday"
        elif value == "no": value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"

    # some out-of-define classification slot values
    if  slot == "restaurant-area" and value in ["stansted airport", "cambridge", "silver street"] or \
        slot == "attraction-area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
        value = "none"

    return value


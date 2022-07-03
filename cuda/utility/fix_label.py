"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

def fix_general_label_error(value):

    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "cen":"centre", 
        "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "scentre":"centre", "town centre":"centre", "in town":"centre", "centre of town":"centre", 
        "east side":"east", "east area":"east", "east part of town":"east", "east of town": "east", "east side of town":"east", "the east":"east", "east part":"east", 
        "west side":"west", "west area":"west", "west part of town":"west", "west of town": "west", "west side of town":"west", "the west":"west", "west part":"west",  
        "south side":"south", "south area":"south", "south part of town":"south", "south of town":"south", "south side of town":"south", "the south":"south", "south part":"south",   
        "north side":"north", "north area":"north", "north part of town":"north", "north of town":"north", "north side of town":"north", "the north":"north", "north part":"north",  
        "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"yes",
        # internet
        "free internet":"yes",
        # star
        "0 star rarting":"none", "ranging from 2 - 4 stars":"none",
        "5-star":"5", "5-stars":"5", 
        "4-star":"4", "4-stars":"4", 
        "3-star":"3", "3-stars":"3",
        "2-star":"2", "2-stars":"2",
        "1-star":"1", "1-stars":"1",
        "0-star":"0",
        # others 
        "y":"yes", "any":"do not care", "n":"no", "does not care":"do not care", "dontcare": "do not care" , "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
        }

    if value in GENERAL_TYPO.keys():
        value = GENERAL_TYPO[value]

    return value


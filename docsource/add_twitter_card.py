import os,sys
import json

index_html_file = sys.argv[1]
twitter_card_json_file = sys.argv[2]

# Twitter Card information
twitter_card_info = json.load(open(twitter_card_json_file))

line_list = None
with open(index_html_file) as fin:
    line_list = fin.readlines()

with open(index_html_file, 'w') as fout:
    card_written = False
    for line in line_list: 
        if line.strip().startswith('<meta') and not card_written:
            # Write twitter cards
            for key in twitter_card_info:
                fout.write('<meta name="%s" content="%s">\n' % (key, twitter_card_info[key]))
            card_written = True
        fout.write(line)


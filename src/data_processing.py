import re
import pandas as pd
import string
def ner_org_heuristic(claim, author_name, date):
    claim = str(claim)
    author_name = str(author_name)
    date = str(date)

    set_potential_org = {'party', 'acttion', 'now', 'blog', 'committee', 'news', 'daily', 'various', 'facebook', 'twitter', 'tweet', 'post', 'posts', 'viral', 'image', 'meme', 'source',
                        'sources', 'association', 'internet', 'chamber', 'republican', 'fund', 'senate', 'fact', 'weekly', 'daily', 'today', 'rumor', 'rumors', 'common', 'last',
                        'for', 'of', 'in', 'from', 'by', 'the', 'update', 'future', 'science', 'foundation', 'reaganwasright', 'nhm-l', 'nation', 'one', 'magazine', 'voice', 'US', 'USA',
                        'initiative', 'organization', 'online', 'and', 'state', 'federation', 'yes', 'no', 'administration', 'true', 'news', 'bureau', 'against', 'conference', 'by', 'national',
                        'mag', 'frontpage','uber','house', 'democrat', 'democrats', 'occupy', 'nrsc', 'ufconly', 'only', 'than', 'attack', 'campaign', 'save', 'my', 'headlines', 'social', 'media',
                        'pundit', 'project', 'nrcc', 'department', 'csc', 'office', 'progress', 'center', 'fund', 'security', 'society', 'other', 'party', 'defense', 'district', 'afscme', 'naral',
                        'school', 'government', 'politico', 'number', 'make', 'border', 'national', 'email', 'mail', 'user', 'council', 'freedomworkers', 'freedom', 'club', 'infowars', 'information',
                        'report', 'quote', 'wisconsin', 'ad', 'network', 'democtratic-npl', 'top', 'press', 'divest', 'times', 'portal', 'news', 'info', 'program', 'one', 'strong', 'united',
                        'oxfam', 'republicans', 'union', 'vote', 'revolution', 'now', 'people', 'on', 'snapchat', 'multiple', 'website', 'websites'}
    set_potential_url = {'http', '.co', '.com', '.live', '.us', '.info', '.xyz', '.net', '.in'}
    new_claim = ''
    author_name_words = author_name.lower().split() #clean extra spaces beforehand as well
    if claim.split()[0].lower() in ['says', 'said']:
        new_claim = author_name+' '+claim+' on '+date
    else:
        if len(author_name_words)==1:
            if any([url_part in author_name_words[0] for url_part in set_potential_url]):
                new_claim = author_name + ' reported that '+claim+' on '+date
            else:
                for word in author_name_words:
                    if word in set_potential_org:
                        new_claim = author_name + ' reported that '+claim+' on '+date
                    else:
                        new_claim = author_name + ' said that '+claim+' on '+date
        elif len(author_name_words)>1:
            for word in author_name_words:
                if word in set_potential_org:
                    new_claim = author_name + ' reported that '+claim+' on '+date
                else:
                    new_claim = author_name + ' said that '+claim+' on '+date

    return new_claim

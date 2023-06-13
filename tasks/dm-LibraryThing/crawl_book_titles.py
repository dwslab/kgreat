"""Use this script to crawl all book titles of the dataset once. Remember to first download the dataset dump."""


import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def _read_relevant_book_ids() -> list:
    actions = {}
    with open('./lthing_data/reviews.txt') as f:
        for line in f:
            data = eval(line[line.find(' = ') + 3:])
            if 'user' not in data or 'work' not in data or 'stars' not in data:
                continue
            actions[(data['user'], int(data['work']))] = float(data['stars'])
    actions = [(user, work, rating) for (user, work), rating in actions.items()]
    actions = pd.DataFrame(data=actions, columns=['user_id', 'item_id', 'rating'])

    actions['item_count'] = actions['item_id'].map(actions['item_id'].value_counts())
    actions['user_count'] = actions['user_id'].map(actions['user_id'].value_counts())
    # remove most popular items (top 1%)
    one_percent_of_items = int(actions['item_id'].nunique() * 0.01)
    top_items = actions['item_id'].value_counts().nlargest(n=one_percent_of_items).index.values
    actions = actions[~actions['item_id'].isin(top_items)]
    # remove items and users with too few ratings (less than five)
    actions = actions[(actions['item_count'] >= 5) & (actions['user_count'] >= 5)]
    actions = actions.drop(columns=['item_count', 'user_count'])
    # return relevant book identifiers
    return list(actions['item_id'].astype(int).unique())


def _crawl_book_title(lt_id: str) -> str:
    response = requests.get(f"https://www.librarything.com/work/{lt_id}")
    soup = BeautifulSoup(response.text, 'html.parser')
    headsummary_element = soup.find('div', class_='headsummary')
    if headsummary_element is None:
        return lt_id
    title_element = headsummary_element.find('h1')
    return title_element.text.strip() if title_element else lt_id


if __name__ == '__main__':
    book_ids = _read_relevant_book_ids()
    book_titles = [_crawl_book_title(book_id) for book_id in tqdm(book_ids)]
    pd.DataFrame({'item_id': book_ids, 'label': book_titles}).to_csv('./books.tsv', sep='\t', index=False)

import codecademylib3_seaborn
from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = 'https://content.codecademy.com/courses/beautifulsoup/cacao/index.html'

webpage_response = requests.get(url)

webpage = webpage_response.content

soup = BeautifulSoup(webpage, "html.parser")

ratings = []

for rating in soup.find_all(attrs={'class':'Rating'}):
  if rating.get_text() == 'Rating':
    pass
  else:
    ratings.append(float(rating.get_text()))

plt.hist(ratings)
plt.show()

companies = []
company_tags = soup.select('.Company')

for td in company_tags[1:]:
  companies.append(td.get_text())

cacao_df = pd.DataFrame({"Company": companies, "Rating": ratings})

mean_vals = cacao_df.groupby("Company").Rating.mean()
ten_best = mean_vals.nlargest(10)

cocao_percents = []
cocao_percent_tags = soup.select('.CocoaPercent')

for td in cocao_percent_tags[1:]:
  percent = float(td.get_text().strip('%'))
  cocao_percents.append(percent)

print(cocao_percents)








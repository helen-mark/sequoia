
KEYWORDS=['Швея',
'Шлифовщик',
'Стирщик',
'Техник',
'Технолог',
'Упаковщик',
'Монтажник',
'Наладчик',
'Специалист по стирке',
'Комплектовщик',
'Комплектовщик-сортировщик',
'Контролер',
'Маляр',
'Маркировщик',
'Мастер',
'Оператор ПК',
'Оператор котельной',
'Оператор прачечной',
'Оператор склада',
'Оператор стиральных машин', 
'Оператор машинной стирки',                                            
'Сборщик светильников',
'Светотехник',
'Сортировщик',
'Гладильщик',
'нормировщик',
'станочник',
'столяр',
'кладовщик',
'оператор прачечной',
'стропальщик',
'укладчик',
'фрезеровщик',
'вальцовщик',
'монтажник',
'Специалист по обслуживанию промышленного оборудования',
'Младший инженер-конструктор',
'оператор плазменной резки',
'Оператор поточно-автоматической линии',
'рабочий',
'оператор линии',
'оператор уборки',
'изготовитель',
'диспетчер',
'специалист обслуживанию']

KEYWORDS = ['инженер по качеству',
'инженер технолог',
'Инженер-технолог',
'инженер по охране труда',
'инженер контроля',
'инженер-испытатель',
'инженер проектировщик',
'Инженер-проектировщик',
'инженер-конструктор',
'инженер по системам',
'Инженер конструктор',
'инженер управления',
'инженер ОТК'
]

KEYWORDS = keywords = [
    "инспектор по кадрам",
    "менеджер по персоналу",
    "менеджер по подбору",
    "специалист по кадрам",
    "специалист по кадровому",
    "3D-аниматор",
    "SMM-менеджер",
    "Контент-менеджер",
    "Графический дизайнер",
    "Интернет-маркетолог",
    "Светодизайнер",
    "Видеооператор",
    "Видео-фотооператор",
    "Графический+дизайнер",
    "Архитектор-дизайнер",
    "Дизайнер-визуализатор",
    "Интернет-маркетолог",
    "Менеджер по маркетинговым коммуникациям",
    "Менеджер по маркетингу и рекламе",
    "Помощник SMM-менеджера",
    "Светодизайнер",
    "Специалист по маркетингу",
    "Специалист по торговому маркетингу",
    "Специалист по связям с общественностью",
    "Менеджер проектов",
    "Менеджер отдела сервиса",
    "менеджер клиентов",
    "менеджер сервиса",
    "менеджер по качеству",
    "менеджер по проектам",
    "менеджер по отчетности",
    "менеджер по снабжению",
    "менеджер региона",
    "менеджер по закупкам",
    "менеджер по обследованиям",
    "менеджер по работе",
    "менеджер по развитию",
    "менеджер по строительству",
    "оператор ПК",
    "Торговый представитель",
    "Ведущий менеджер активных продаж",
    "Менеджер по продажам",
    "Менеджер отдела продаж",
    "координатор отдела продаж",
    "Помощник менеджера по продажам",
    "Администратор отдела продаж",
    "Начальник отдела по развитию продаж",
    "Руководитель отдела продаж",
    "Специалист технической поддержки",
    "Офис-менеджер",
    "Переводчик",
    "Помощник менеджера по закупкам",
    "Секретарь-референт",
    "Специалист по документообороту",
    "Специалист по охране труда",
    "Специалист по работе с документами",
    "Специалист по работе с документацией",
    "Специалист по административно-хозяйственной деятельности",
    "Тренинг-менеджер",
    "Старший менеджер по административно-хозяйственной части",
    "Ведущий специалист",
    "Диспетчер",
    "Консультант по вопросам интеграции корпоративных стандартов",
    "Координатор",
    "Куратор",
    "Курьер",
    "Лаборант",
    "Лаборант электротехнической лаборатории",
    "Специалист по доставке",
    "Стажер",
    "менеджер по административно",
    "аналитик",
    "бухгалтер",
    "Бизнес-ассистент",
    "Бизнес-тренер",
    "кассир",
    "экономист",
    "Юрисконсульт",
    "Юрист",
    "Финансовый контролер",
    "Ведущий специалист тендерного отдела",
    "Специалист по взаиморасчетам",
    "менеджер по взаиморасчетам",
    "Ревизор",
    "Контролер-ревизор"
]

# KEYWORDS=['специалист по логистике',
# 'старший водитель экспедитор',
# 'специалист+перевозок',
# 'логист',
# 'менеджер по логистике'
# ]

# KEYWORDS=['медицинская сестра', "медсестра", "фельдшер"]
# KEYWORDS=['Сопроводитель грузов',
# 'сторож',
# 'Кладовщик',
# 'Подсобный рабочий',
# 'Разнорабочий',
# 'Грузчик',
# 'Специалист по складскому учету',
# 'уборщица',
# 'Специалист по уборке помещений',
# 'оператор службы уборки'
# ]

import requests
import json
import pandas as pd
from time import sleep
import glob
from tqdm import tqdm  # для красивого прогресс-бара (установите через pip install tqdm)

# Настройки
# KEYWORDS = [
#     'фрезеровщик', 'токарь', 'электрогазосварщик', 'сварщик', 'слесарь',
#     'изолировщик', 'ремонтник', 'газорезчик', 'механик', 'электромонтер',
#     'электромонтажник'
# ]

KEYWORDS=['начальник цеха',
'начальник отдела',
'начальник склада',
'начальник участка',
'руководитель отдела',
'руководитель подразделения',
'руководитель группы',
'руководитель направления',
'руководитель проектов',
'руководитель казначейства',
'руководитель проекта',
'руководитель филиала',
'заместитель начальника',
'заместитель руководителя',
'начальник производства',
'бригадир',
'главный диспетчер',
'заведующий',
'управляющий',
'старший координатор'
]

CITIES = pd.read_excel('data_raw/salary_by_cities.xlsx', sheet_name='Sheet2')['city'].tolist()  # ["Новомосковск", "Белореченск", "Невинномысск", "Кингисепп", "Москва", "Новосибирск", "Санкт-Петербург", "Нижний Новгород", "Екатеринбург", "Краснодар"]
OUTPUT_FILE = "hh_results_chief.json"


# Получаем ID города по названию (оставим на случай, если API будет корректно работать)
def get_city_id(city_name):
    cities = requests.get("https://api.hh.ru/areas").json()
    for country in cities:
        for region in country["areas"]:
            for city in region["areas"]:
                if city["name"].lower() == city_name.lower():
                    return city["id"]
    return None


# Проверяем, что вакансия действительно относится к нужному городу
def is_vacancy_in_city(vacancy, city_name):
    # Проверяем в адресе
    if vacancy.get('address') and vacancy['address'].get('city'):
        if city_name.lower() in vacancy['address']['city'].lower():
            return True

    # Проверяем в названии города в описании
    # if vacancy.get('area') and vacancy['area'].get('name'):
    #     if city_name.lower() in vacancy['area']['name'].lower():
    #         return True

    # Проверяем в названии вакансии или описании (на всякий случай)
    #if city_name.lower() in vacancy.get('name', '').lower():
    #return True

    return False


# Сбор всех вакансий по городу (для всех ключевых слов)
def fetch_all_vacancies(city_name):
    # city_id = get_city_id(city_name)
    # if not city_id:
    #     print(f"Город {city_name} не найден!")
    #     return []

    all_vacancies = []

    for keyword in KEYWORDS:
        page = 0
        pages_to_process = 1  # Начинаем с предположения, что есть хотя бы 1 страница

        while page < pages_to_process:
            params = {
                "text": keyword,
              #  "area": city_id,
                "per_page": 100,
                "page": page
            }

            try:
                response = requests.get("https://api.hh.ru/vacancies", params=params)
                response.raise_for_status()
                data = response.json()

                # Обновляем общее количество страниц
                pages_to_process = data.get("pages", 1)

                # Фильтруем вакансии по городу
                filtered = [v for v in data.get("items", []) if is_vacancy_in_city(v, city_name)]
                all_vacancies.extend(filtered)

                print(
                    f"Обработано {len(filtered)} вакансий по '{keyword}' (страница {page + 1}/{pages_to_process})")

                page += 1
                sleep(0.5)  # Задержка между запросами

            except Exception as e:
                print(f"Ошибка при запросе по '{keyword}', страница {page}: {e}")
                break

    return all_vacancies


# Расчёт средней зарплаты и количества вакансий (оставляем без изменений)
def calculate_stats(vacancies):
    salaries = []
    for vacancy in vacancies:
        salary = vacancy.get("salary")
        if salary:
            from_salary = salary.get("from")
            to_salary = salary.get("to")

            if from_salary and to_salary:
                avg = (from_salary + to_salary) / 2
                salaries.append(avg)
            elif from_salary:
                salaries.append(from_salary)
            elif to_salary:
                salaries.append(to_salary)

    average_salary = sum(salaries) / len(salaries) if salaries else 0
    return {
        "total_vacancies": len(vacancies),
        "average_salary": round(average_salary, 2),
        "vacancies_with_salary": len(salaries)
    }


# Сохранение результатов в файл (дописываем)
def save_results(city_name, stats, filename=OUTPUT_FILE):
    try:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}

        data[city_name] = stats

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Результаты для {city_name} сохранены в {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")


# Основной цикл
if __name__ == "__main__":
    # for city in CITIES:
    #     print(f"\nОбрабатываю город: {city}")
    #     vacancies = fetch_all_vacancies(city)
    #     stats = calculate_stats(vacancies)
    #     print(f"Итого в {city}:")
    #     print(f"- Всего вакансий: {stats['total_vacancies']}")
    #     print(f"- Вакансий с указанной зарплатой: {stats['vacancies_with_salary']}")
    #     print(f"- Средняя з/п: {stats['average_salary']} руб.")
    #     save_results(city, stats)
    #
    df = pd.read_excel('data_raw/химик1.xlsx', sheet_name='Основные данные')
    # Assuming df is your DataFrame
    population_map = {
        'Белореченск': 54190,
        'Невинномысск': 113112,
        'Новомосковск': 115938,
        'г. Санкт-Петербург': 5652922,
        'г. Москва': 13274285,
        'Кингисепп': 48355,
        'Алматы': 2211198,
        'Алма-Ата': 2211198  # alternative name
    }

    df['City population'] = df['City'].apply(lambda x: population_map.get(x, 1000000))

    df1 = pd.read_excel('job_categories.xlsx')
    # Создаем словарь соответствий "должность → категория"
    job_to_category = df1.set_index('должность')['категория'].to_dict()

    # Заменяем значения в df2['Job title'], если они есть в словаре
    df['Job title'] = df['Job title'].map(job_to_category).fillna(df['Job title'])

    df.to_excel('data_raw/химик_осн.xlsx')


    # Step 1: Collect all JSON files matching the pattern
    # json_files = glob.glob('hh_results_*.json')
    #
    # # Initialize an empty DataFrame with 'city' as the index
    # df = pd.DataFrame()
    #
    # for file in json_files:
    #     # Extract profession name (e.g., 'driver' from 'hh_results_driver.json')
    #     profession = file.split('_')[-1].replace('.json', '')
    #
    #     # Load JSON data
    #     with open(file, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #
    #     # Convert JSON to a temporary DataFrame
    #     temp_df = pd.DataFrame.from_dict(data, orient='index')
    #     temp_df['city'] = temp_df.index  # Add city column
    #
    #     # Keep only 'city' and 'total_vacancies' (rename column to profession)
    #     temp_df = temp_df[['city', 'average_salary']]
    #     temp_df = temp_df.rename(columns={'average_salary': profession})
    #
    #     # Merge into the main DataFrame
    #     if df.empty:
    #         df = temp_df
    #     else:
    #         df = pd.merge(df, temp_df, on='city', how='outer')
    #
    # # Fill NaN with 0 (if a city has no vacancies for a profession)
    # df = df.fillna(0)
    #
    # # Sort cities alphabetically (optional)
    # df = df.sort_values('city')
    #
    # # Export to Excel
    # out_file = 'average_salary.xlsx'
    # df.to_excel(out_file, index=False)
    # print(f"Excel file {out_file} created successfully!")
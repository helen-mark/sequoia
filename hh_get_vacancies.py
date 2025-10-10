
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
#
# KEYWORDS = ['инженер по качеству',
# 'инженер технолог',
# 'Инженер-технолог',
# 'инженер по охране труда',
# 'инженер контроля',
# 'инженер-испытатель',
# 'инженер проектировщик',
# 'Инженер-проектировщик',
# 'инженер-конструктор',
# 'инженер по системам',
# 'Инженер конструктор',
# 'инженер управления',
# 'инженер ОТК'
# ]
#
# KEYWORDS = [
#     "инспектор по кадрам",
#     "менеджер по персоналу",
#     "менеджер по подбору",
#     "специалист по кадрам",
#     "специалист по кадровому",
#     "3D-аниматор",
#     "SMM-менеджер",
#     "Контент-менеджер",
#     "Графический дизайнер",
#     "Интернет-маркетолог",
#     "Светодизайнер",
#     "Видеооператор",
#     "Видео-фотооператор",
#     "Графический+дизайнер",
#     "Архитектор-дизайнер",
#     "Дизайнер-визуализатор",
#     "Интернет-маркетолог",
#     "Менеджер по маркетинговым коммуникациям",
#     "Менеджер по маркетингу и рекламе",
#     "Помощник SMM-менеджера",
#     "Светодизайнер",
#     "Специалист по маркетингу",
#     "Специалист по торговому маркетингу",
#     "Специалист по связям с общественностью",
#     "Менеджер проектов",
#     "Менеджер отдела сервиса",
#     "менеджер клиентов",
#     "менеджер сервиса",
#     "менеджер по качеству",
#     "менеджер по проектам",
#     "менеджер по отчетности",
#     "менеджер по снабжению",
#     "менеджер региона",
#     "менеджер по закупкам",
#     "менеджер по обследованиям",
#     "менеджер по работе",
#     "менеджер по развитию",
#     "менеджер по строительству",
#     "оператор ПК",
#     "Торговый представитель",
#     "Ведущий менеджер активных продаж",
#     "Менеджер по продажам",
#     "Менеджер отдела продаж",
#     "координатор отдела продаж",
#     "Помощник менеджера по продажам",
#     "Администратор отдела продаж",
#     "Начальник отдела по развитию продаж",
#     "Руководитель отдела продаж",
#     "Специалист технической поддержки",
#     "Офис-менеджер",
#     "Переводчик",
#     "Помощник менеджера по закупкам",
#     "Секретарь-референт",
#     "Специалист по документообороту",
#     "Специалист по охране труда",
#     "Специалист по работе с документами",
#     "Специалист по работе с документацией",
#     "Специалист по административно-хозяйственной деятельности",
#     "Тренинг-менеджер",
#     "Старший менеджер по административно-хозяйственной части",
#     "Ведущий специалист",
#     "Диспетчер",
#     "Консультант по вопросам интеграции корпоративных стандартов",
#     "Координатор",
#     "Куратор",
#     "Курьер",
#     "Лаборант",
#     "Лаборант электротехнической лаборатории",
#     "Специалист по доставке",
#     "Стажер",
#     "менеджер по административно",
#     "аналитик",
#     "бухгалтер",
#     "Бизнес-ассистент",
#     "Бизнес-тренер",
#     "кассир",
#     "экономист",
#     "Юрисконсульт",
#     "Юрист",
#     "Финансовый контролер",
#     "Ведущий специалист тендерного отдела",
#     "Специалист по взаиморасчетам",
#     "менеджер по взаиморасчетам",
#     "Ревизор",
#     "Контролер-ревизор"
# ]

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


# Настройки
# KEYWORDS = [
#     'фрезеровщик', 'токарь', 'электрогазосварщик', 'сварщик', 'слесарь',
#     'изолировщик', 'ремонтник', 'газорезчик', 'механик', 'электромонтер',
#     'электромонтажник', 'электросварщик', 'монтер'
# ]
#
# KEYWORDS=['начальник цеха',
# 'начальник отдела',
# 'начальник склада',
# 'начальник участка',
# 'руководитель отдела',
# 'руководитель подразделения',
# 'руководитель группы',
# 'руководитель направления',
# 'руководитель проектов',
# 'руководитель казначейства',
# 'руководитель проекта',
# 'руководитель филиала',
# 'заместитель начальника',
# 'заместитель руководителя',
# 'начальник производства',
# 'бригадир',
# 'главный диспетчер',
# 'заведующий',
# 'управляющий',
# 'старший координатор'
# ]

# KEYWORDS = ['водитель автотранспорта', 'личный водитель', 'водитель экспедитор', 'водитель грузового', 'водитель магистрального', 'водитель автобуса',
#             'персональный водитель', 'водитель микроавтобуса', 'водитель автомобиля']

CITIES = pd.read_excel('data_raw/salary_by_cities.xlsx', sheet_name='Sheet2')['city'].tolist()  # ["Новомосковск", "Белореченск", "Невинномысск", "Кингисепп", "Москва", "Новосибирск", "Санкт-Петербург", "Нижний Новгород", "Екатеринбург", "Краснодар"]
OUTPUT_FILE = "hh_results_driver_u.json"


# Получаем ID города по названию (оставим на случай, если API будет корректно работать)
def get_city_id(city_name):
    cities = requests.get("https://api.hh.ru/areas").json()
    for country in cities:
        for region in country["areas"]:
            for city in region["areas"]:
                if city["name"].lower() == city_name.lower():
                    return city["id"]
    return None


def convert_shift_to_monthly(from_salary, to_salary, fly_in_fly_out_duration):
    """
    ПРАВИЛЬНО преобразует вахтенную зарплату в месячную
    """
    if not fly_in_fly_out_duration:
        return None

    # Извлекаем числовые значения продолжительности вахты из ID
    shift_durations = []
    for duration in fly_in_fly_out_duration:
        duration_id = duration.get('id', '')
        if duration_id.startswith('DAYS_'):
            try:
                days = int(duration_id.replace('DAYS_', ''))
                shift_durations.append(days)
            except ValueError:
                continue

    if not shift_durations:
        return None

    # Сортируем продолжительности
    shift_durations.sort()
    min_duration = shift_durations[0]
    max_duration = shift_durations[-1]

    # ПРАВИЛЬНАЯ логика:
    if from_salary and to_salary:
        # Обе границы: from -> к минимальной продолжительности, to -> к максимальной
        converted_from = from_salary * (30 / min_duration) if from_salary else None
        converted_to = to_salary * (30 / max_duration) if to_salary else None
        print(f"Обе границы: {from_salary}-{to_salary} за вахту")
        print(f"from({from_salary}) → min({min_duration}д) = {converted_from:.0f}/мес")
        print(f"to({to_salary}) → max({max_duration}д) = {converted_to:.0f}/мес")

    elif from_salary and not to_salary:
        # Только нижняя граница - берем МИНИМАЛЬНУЮ продолжительность
        converted_from = from_salary * (30 / min_duration)
        converted_to = None
        print(f"Только from: {from_salary} за вахту → min({min_duration}д) = {converted_from:.0f}/мес")

    elif to_salary and not from_salary:
        # Только верхняя граница - берем МАКСИМАЛЬНУЮ продолжительность
        converted_from = None
        converted_to = to_salary * (30 / max_duration)
        print(f"Только to: {to_salary} за вахту → max({max_duration}д) = {converted_to:.0f}/мес")

    else:
        return None

    return (converted_from, converted_to)

# Проверяем, что вакансия действительно относится к нужному городу
def is_vacancy_in_city(vacancy, city_name):
    # Проверяем в адресе
    if vacancy.get('address') and vacancy['address'].get('city'):
        if city_name.lower() in vacancy['address']['city'].lower():
            return True

    # Проверяем в названии города в описании
    if vacancy.get('area') and vacancy['area'].get('name'):
        if city_name.lower() in vacancy['area']['name'].lower():
            return True

    # Проверяем в названии вакансии или описании (на всякий случай)
    #if city_name.lower() in vacancy.get('name', '').lower():
    #return True

    return False


# Сбор всех вакансий по городу (для всех ключевых слов) с проверкой уникальности
def fetch_all_vacancies(city_name):
    all_vacancies = []
    seen_ids = set()  # Для отслеживания уникальных ID вакансий

    for keyword in KEYWORDS:
        page = 0
        pages_to_process = 1

        while page < pages_to_process:
            params = {
                "text": keyword,
                "per_page": 100,
                "page": page
            }

            try:
                response = requests.get("https://api.hh.ru/vacancies", params=params)
                response.raise_for_status()
                data = response.json()
                pages_to_process = data.get("pages", 1)

                # Фильтруем вакансии по городу и уникальности
                filtered = []
                for v in data.get("items", []):
                    if (is_vacancy_in_city(v, city_name) and
                            v.get('id') not in seen_ids):
                        seen_ids.add(v['id'])
                        filtered.append(v)
                        try:
                            if v.get('salary_range').get('mode').get('id') != 'MONTH':
                                print(v)
                            elif v.get('salary_range').get('mode').get('id') == 'MONTH':
                                print('month')
                            else:
                                print(v.get('salary_range').get('mode'))
                        except Exception as e:
                            print(e)

                all_vacancies.extend(filtered)
                print(
                    f"Обработано {len(filtered)} уникальных вакансий по '{keyword}' (страница {page + 1}/{pages_to_process})")
                page += 1
                sleep(0.5)

            except Exception as e:
                print(f"Ошибка при запросе по '{keyword}', страница {page}: {e}")
                break

    return all_vacancies


# Расчёт средней зарплаты и количества вакансий с проверкой уникальности
def calculate_stats(vacancies):
    # Разделяем вакансии на три группы
    group_a = []  # Только "от" (from_salary)
    group_b = []  # Только "до" (to_salary)
    group_c = []  # Полная вилка (from_salary и to_salary)

    seen_ids = set()
    unique_vacancies = 0

    # Шаг 1: Сбор и категоризация данных с учетом gross и вахтенного графика
    for vacancy in vacancies:
            vacancy_id = vacancy.get('id')
            if vacancy_id and vacancy_id not in seen_ids:
                seen_ids.add(vacancy_id)
                unique_vacancies += 1

                salary = vacancy.get("salary")
                if salary:
                    # Получаем информацию о периоде оплаты из salary_range
                    salary_range = vacancy.get('salary_range', {})
                    try:
                        salary_mode = salary_range.get('mode', {}) if salary_range else salary.get('mode', {})
                    except Exception as e:
                        print('could not fetch mode; assuming "MONTH"')
                        salary_mode = 'MONTH'

                    if salary_mode not in ['MONTH', 'FLY_IN_FLY_OUT']:  # monthly payment or BAXTA
                        continue  # we can't process other modes yet


                    from_salary = salary.get("from")
                    to_salary = salary.get("to")
                    gross = salary.get("gross", True)

                    # Приводим к "на руки" (нетто), если указано gross=True
                    if gross:
                        tax_coefficient = 0.87
                        if from_salary:
                            from_salary = from_salary * tax_coefficient
                        if to_salary:
                            to_salary = to_salary * tax_coefficient

                    # Проверяем, является ли зарплата вахтенной
                    if salary_mode and salary_mode.get('id') == 'FLY_IN_FLY_OUT':
                        # Получаем возможные продолжительности вахты
                        fly_in_fly_out_duration = vacancy.get('fly_in_fly_out_duration', [])

                        # Преобразуем вахтенную зарплату в месячную
                        if from_salary or to_salary:
                            monthly_salaries = convert_shift_to_monthly(
                                from_salary, to_salary, fly_in_fly_out_duration
                            )
                            if monthly_salaries:
                                from_salary, to_salary = monthly_salaries

                    if from_salary and to_salary:
                        group_c.append((from_salary, to_salary))
                    elif from_salary:
                        group_a.append(from_salary)
                    elif to_salary:
                        group_b.append(to_salary)

    # Шаг 2: Обработка каждой группы
    calculated_salaries = []

    if group_c:
        # Основной сценарий: есть полные вилки
        # Сначала добавляем средние по полным вилкам
        for from_sal, to_sal in group_c:
            avg = (from_sal + to_sal) / 2
            calculated_salaries.append(avg)

        # Вычисляем коэффициенты для групп A и B
        ratios = []
        for from_sal, to_sal in group_c:
            if from_sal > 0:  # Избегаем деления на ноль
                ratio = to_sal / from_sal
                ratios.append(ratio)

        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0

        # Обрабатываем группу A (только "от")
        for from_sal in group_a:
            estimated_to = from_sal * avg_ratio
            estimated_avg = (from_sal + estimated_to) / 2
            calculated_salaries.append(estimated_avg)

        # Обрабатываем группу B (только "до")
        for to_sal in group_b:
            estimated_from = to_sal / avg_ratio
            estimated_avg = (estimated_from + to_sal) / 2
            calculated_salaries.append(estimated_avg)

    else:
        # Fallback-сценарий: нет полных вилок
        median_from = 0
        median_to = 0

        if group_a:
            sorted_a = sorted(group_a)
            n_a = len(sorted_a)
            if n_a % 2 == 1:
                median_from = sorted_a[n_a // 2]
            else:
                median_from = (sorted_a[n_a // 2 - 1] + sorted_a[n_a // 2]) / 2

        if group_b:
            sorted_b = sorted(group_b)
            n_b = len(sorted_b)
            if n_b % 2 == 1:
                median_to = sorted_b[n_b // 2]
            else:
                median_to = (sorted_b[n_b // 2 - 1] + sorted_b[n_b // 2]) / 2

        if group_a and group_b:
            fallback_avg = (median_from + median_to) / 2
            calculated_salaries.append(fallback_avg)
            fallback_method = "median_from_and_median_to_avg"
        elif group_a:
            calculated_salaries.append(median_from)
            fallback_method = "median_from_only"
        elif group_b:
            calculated_salaries.append(median_to)
            fallback_method = "median_to_only"
        else:
            fallback_method = "no_data"

    # Шаг 3: Финальный расчет
    if calculated_salaries:
        sorted_salaries = sorted(calculated_salaries)
        n = len(sorted_salaries)

        if n % 2 == 1:
            median_salary = sorted_salaries[n // 2]
        else:
            median_salary = (sorted_salaries[n // 2 - 1] + sorted_salaries[n // 2]) / 2
    else:
        median_salary = 0

    result = {
        "total_vacancies": unique_vacancies,
        "median_salary": round(median_salary, 2),
        "vacancies_with_salary": len(calculated_salaries),
        "group_a_count": len(group_a),
        "group_b_count": len(group_b),
        "group_c_count": len(group_c),
        "salary_type": "net"  # Теперь все зарплаты приведены к "на руки"
    }

    # Добавляем информацию о методе расчета
    if group_c:
        result["calculation_method"] = "full_range_with_ratio"
        result["avg_ratio"] = round(avg_ratio, 3) if group_c else 0
    else:
        result["calculation_method"] = fallback_method
        if group_a:
            result["median_from"] = round(median_from, 2)
        if group_b:
            result["median_to"] = round(median_to, 2)

    return result


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
    for city in CITIES:
        print(f"\nОбрабатываю город: {city}")
        if city == 'Арлан':
            continue
        vacancies = fetch_all_vacancies(city)
        stats = calculate_stats(vacancies)
        print(f"Итого в {city}:")
        print(f"- Всего вакансий: {stats['total_vacancies']}")
        print(f"- Вакансий с указанной зарплатой: {stats['vacancies_with_salary']}")
        print(f"- Средняя з/п: {stats['median_salary']} руб.")
        save_results(city, stats)

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

    # df['City population'] = df['City'].apply(lambda x: population_map.get(x, 1000000))
    #
    # df1 = pd.read_excel('job_categories.xlsx')
    # # Создаем словарь соответствий "должность → категория"
    # job_to_category = df1.set_index('должность')['категория'].to_dict()
    #
    # # Заменяем значения в df2['Job title'], если они есть в словаре
    # df['Job title'] = df['Job title'].map(job_to_category).fillna(df['Job title'])
    #
    # df.to_excel('data_raw/химик_осн.xlsx')


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
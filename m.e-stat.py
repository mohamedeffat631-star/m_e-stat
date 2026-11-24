import os
import math
import pandas as pd
from itertools import combinations

try:
    from scipy.stats import f as f_distribution
    from scipy.stats import t as t_distribution
    from scipy.stats import studentized_range
except ImportError:
    f_distribution = None
    t_distribution = None
    studentized_range = None


# ==========================================
# وظائف عامة
# ==========================================

def ask(prompt):
    return input(prompt + " ").strip()


def choose_column(df, label):
    print("\nأعمدة الملف:")

    for i, col in enumerate(df.columns, start=1):
        print(f"{i} - {col}")

    while True:
        choice = ask(f"اختر رقم عمود {label}:")

        try:
            index = int(choice) - 1

            if 0 <= index < len(df.columns):
                return df.columns[index]

        except ValueError:
            pass

        print("اختيار غير صحيح، حاول مرة أخرى.")


def load_file():
    path = ask("أدخل المسار الكامل لملف البيانات:")

    if not os.path.isfile(path):
        print("لم يتم العثور على الملف.")
        return None

    try:
        extension = os.path.splitext(path)[1].lower()

        if extension == ".csv":
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="cp1256")

        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(path)

        else:
            print("نوع الملف غير مدعوم.")
            return None

        if df.empty:
            print("الملف لا يحتوي على بيانات.")
            return None

        print("\nتم تحميل الملف بنجاح.")
        print("عدد الصفوف:", len(df))
        print("عدد الأعمدة:", len(df.columns))

        return df

    except Exception as error:
        print("حدث خطأ أثناء قراءة الملف:")
        print(error)
        return None


# ==========================================
# اختبارات المقارنة بين المتوسطات
# ==========================================

def make_letters(number):
    result = ""

    while number >= 0:
        result = chr(number % 26 + 65) + result
        number = number // 26 - 1

    return result


def compact_letters(names, nonsignificant_pairs):
    groups = []

    for name in names:
        placed = False

        for group in groups:
            if all(
                tuple(sorted((name, other))) in nonsignificant_pairs
                for other in group
            ):
                group.append(name)
                placed = True

        if not placed:
            groups.append([name])

    letters = {name: "" for name in names}

    for index, group in enumerate(groups):
        letter = make_letters(index)

        for name in group:
            letters[name] += letter

    return letters


def duncan_test(means, mse, df_error, replicates, alpha):
    if studentized_range is None:
        return None

    names = list(means.keys())

    ordered = sorted(
        means.items(),
        key=lambda item: item[1],
        reverse=True
    )

    comparisons = []
    nonsignificant_pairs = set()

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):

            name1, mean1 = ordered[i]
            name2, mean2 = ordered[j]

            span = j - i + 1
            difference = abs(mean1 - mean2)

            if mse == 0:
                critical_range = 0
                significant = difference > 0
                p_value = 0.0 if significant else 1.0

            else:
                adjusted_alpha = 1 - (1 - alpha) ** (span - 1)

                critical_q = studentized_range.ppf(
                    1 - adjusted_alpha,
                    span,
                    df_error
                )

                critical_range = (
                    critical_q * math.sqrt(mse / replicates)
                )

                significant = difference > critical_range

                q_observed = (
                    difference / math.sqrt(mse / replicates)
                )

                p_value = studentized_range.sf(
                    q_observed,
                    span,
                    df_error
                )

            if not significant:
                nonsignificant_pairs.add(
                    tuple(sorted((name1, name2)))
                )

            comparisons.append({
                "name1": name1,
                "name2": name2,
                "difference": difference,
                "p_value": p_value,
                "significant": significant
            })

    return {
        "comparisons": comparisons,
        "letters": compact_letters(
            names,
            nonsignificant_pairs
        )
    }


def tukey_test(means, mse, df_error, replicates):
    if studentized_range is None:
        return None

    k = len(means)

    if k < 2:
        return None

    if mse == 0:
        hsd_5 = 0
        hsd_1 = 0
    else:
        standard_error = math.sqrt(mse / replicates)

        hsd_5 = (
            studentized_range.ppf(0.95, k, df_error)
            * standard_error
        )

        hsd_1 = (
            studentized_range.ppf(0.99, k, df_error)
            * standard_error
        )

    comparisons = []

    for (name1, mean1), (name2, mean2) in combinations(
        means.items(), 2
    ):

        difference = abs(mean1 - mean2)

        if mse == 0:
            p_value = 0.0 if difference > 0 else 1.0
            significant_5 = difference > 0
            significant_1 = difference > 0

        else:
            q_observed = difference / math.sqrt(mse / replicates)

            p_value = studentized_range.sf(
                q_observed,
                k,
                df_error
            )

            significant_5 = difference > hsd_5
            significant_1 = difference > hsd_1

        comparisons.append({
            "name1": name1,
            "name2": name2,
            "difference": difference,
            "p_value": p_value,
            "significant_5": significant_5,
            "significant_1": significant_1
        })

    return {
        "hsd_5": hsd_5,
        "hsd_1": hsd_1,
        "comparisons": comparisons
    }


def scheffe_test(means, mse, df_error, replicates):
    if f_distribution is None:
        return None

    k = len(means)

    if k < 2:
        return None

    df_scheffe = k - 1

    if mse == 0:
        critical_5 = 0
        critical_1 = 0
    else:
        f_5 = f_distribution.ppf(
            0.95,
            df_scheffe,
            df_error
        )

        f_1 = f_distribution.ppf(
            0.99,
            df_scheffe,
            df_error
        )

        critical_5 = math.sqrt(
            df_scheffe * f_5 * mse * 2 / replicates
        )

        critical_1 = math.sqrt(
            df_scheffe * f_1 * mse * 2 / replicates
        )

    comparisons = []

    for (name1, mean1), (name2, mean2) in combinations(
        means.items(), 2
    ):

        difference = abs(mean1 - mean2)

        if mse == 0:
            p_value = 0.0 if difference > 0 else 1.0
            significant_5 = difference > 0
            significant_1 = difference > 0

        else:
            contrast_f = (
                difference ** 2
                / (mse * 2 / replicates)
            )

            observed_f = contrast_f / df_scheffe

            p_value = f_distribution.sf(
                observed_f,
                df_scheffe,
                df_error
            )

            significant_5 = difference > critical_5
            significant_1 = difference > critical_1

        comparisons.append({
            "name1": name1,
            "name2": name2,
            "difference": difference,
            "p_value": p_value,
            "significant_5": significant_5,
            "significant_1": significant_1
        })

    return {
        "critical_5": critical_5,
        "critical_1": critical_1,
        "comparisons": comparisons
    }


def lsd_test(means, mse, df_error, replicates):
    if t_distribution is None:
        return None

    if mse == 0:
        return {
            "lsd_5": 0,
            "lsd_1": 0
        }

    lsd_5 = (
        t_distribution.ppf(0.975, df_error)
        * math.sqrt(2 * mse / replicates)
    )

    lsd_1 = (
        t_distribution.ppf(0.995, df_error)
        * math.sqrt(2 * mse / replicates)
    )

    return {
        "lsd_5": lsd_5,
        "lsd_1": lsd_1
    }


# ==========================================
# تنفيذ اختبارات المقارنة
# ==========================================

def calculate_posthoc(means, mse, df_error, replicates):

    return {
        "lsd": lsd_test(
            means, mse, df_error, replicates
        ),

        "duncan_5": duncan_test(
            means, mse, df_error, replicates, 0.05
        ),

        "duncan_1": duncan_test(
            means, mse, df_error, replicates, 0.01
        ),

        "tukey": tukey_test(
            means, mse, df_error, replicates
        ),

        "scheffe": scheffe_test(
            means, mse, df_error, replicates
        )
    }


# ==========================================
# تحليل تصميم القطاعات الكاملة العشوائية
# RCBD
# ==========================================

def analyze_rcbd(df, treatment, block, response):

    data = df[[treatment, block, response]].copy()

    data[response] = pd.to_numeric(
        data[response],
        errors="coerce"
    )

    if data.isnull().any().any():
        raise ValueError(
            "توجد خلايا فارغة أو قيم غير رقمية."
        )

    treatments = list(data[treatment].unique())
    blocks = list(data[block].unique())

    k = len(treatments)
    r = len(blocks)
    total = len(data)

    if k < 2 or r < 2:
        raise ValueError(
            "يجب وجود معاملتين وقطاعين على الأقل."
        )

    if total != k * r:
        raise ValueError(
            "البيانات لا تمثل تصميم RCBD متوازن."
        )

    counts = pd.crosstab(
        data[treatment],
        data[block]
    )

    if not (counts == 1).all().all():
        raise ValueError(
            "يجب وجود مشاهدة واحدة لكل معاملة داخل كل قطاع."
        )

    grand_mean = data[response].mean()

    treatment_means = data.groupby(
        treatment
    )[response].mean()

    block_means = data.groupby(
        block
    )[response].mean()

    ss_total = (
        (data[response] - grand_mean) ** 2
    ).sum()

    ss_treatment = (
        r * ((treatment_means - grand_mean) ** 2).sum()
    )

    ss_block = (
        k * ((block_means - grand_mean) ** 2).sum()
    )

    ss_error = ss_total - ss_treatment - ss_block

    df_treatment = k - 1
    df_block = r - 1
    df_error = (k - 1) * (r - 1)
    df_total = total - 1

    ms_treatment = ss_treatment / df_treatment
    ms_block = ss_block / df_block
    ms_error = ss_error / df_error

    if ms_error < 0 and abs(ms_error) < 1e-10:
        ms_error = 0

    if ms_error < 0:
        raise ValueError("خطأ في حساب متوسط مربعات الخطأ.")

    if ms_error == 0:
        f_treatment = float("inf") if ms_treatment > 0 else 0
        f_block = float("inf") if ms_block > 0 else 0

        p_treatment = 0.0 if ms_treatment > 0 else 1.0
        p_block = 0.0 if ms_block > 0 else 1.0

    else:
        f_treatment = ms_treatment / ms_error
        f_block = ms_block / ms_error

        f_treatment_p = f_distribution.sf(
            f_treatment, df_treatment, df_error
        )

        f_block_p = f_distribution.sf(
            f_block, df_block, df_error
        )

        p_treatment = f_treatment_p
        p_block = f_block_p

    means = treatment_means.to_dict()

    posthoc = calculate_posthoc(
        means,
        ms_error,
        df_error,
        r
    )

    return {
        "design": "RCBD",
        "means": means,
        "grand_mean": grand_mean,
        "number_of_treatments": k,
        "number_of_blocks": r,
        "anova": {
            "treatment": {
                "ss": ss_treatment,
                "df": df_treatment,
                "ms": ms_treatment,
                "f": f_treatment,
                "p": p_treatment
            },
            "block": {
                "ss": ss_block,
                "df": df_block,
                "ms": ms_block,
                "f": f_block,
                "p": p_block
            },
            "error": {
                "ss": ss_error,
                "df": df_error,
                "ms": ms_error
            },
            "total": {
                "ss": ss_total,
                "df": df_total
            }
        },
        "posthoc": posthoc
    }


# ==========================================
# تحليل تصميم المربع اللاتيني
# Latin Square Design
# ==========================================

def analyze_latin_square(
    df,
    treatment,
    row,
    column,
    response
):

    data = df[
        [treatment, row, column, response]
    ].copy()

    data[response] = pd.to_numeric(
        data[response],
        errors="coerce"
    )

    if data.isnull().any().any():
        raise ValueError(
            "توجد خلايا فارغة أو قيم غير رقمية."
        )

    treatments = list(data[treatment].unique())
    rows = list(data[row].unique())
    columns = list(data[column].unique())

    k = len(treatments)
    n_rows = len(rows)
    n_columns = len(columns)

    if k < 3:
        raise ValueError(
            "يجب أن يحتوي المربع اللاتيني على ثلاثة معاملات على الأقل."
        )

    if not (k == n_rows == n_columns):
        raise ValueError(
            "عدد المعاملات والصفوف والأعمدة يجب أن يكون متساوياً."
        )

    if len(data) != k * k:
        raise ValueError(
            "عدد المشاهدات غير صحيح لهذا المربع اللاتيني."
        )

    # التحقق من عدم تكرار خلية الصف والعمود
    if data.duplicated(subset=[row, column]).any():
        raise ValueError(
            "يوجد تكرار في خلايا الصفوف والأعمدة."
        )

    # التحقق من ظهور كل معاملة مرة واحدة في كل صف
    row_table = pd.crosstab(
        data[row],
        data[treatment]
    )

    if not (row_table == 1).all().all():
        raise ValueError(
            "كل معاملة يجب أن تظهر مرة واحدة فقط في كل صف."
        )

    # التحقق من ظهور كل معاملة مرة واحدة في كل عمود
    column_table = pd.crosstab(
        data[column],
        data[treatment]
    )

    if not (column_table == 1).all().all():
        raise ValueError(
            "كل معاملة يجب أن تظهر مرة واحدة فقط في كل عمود."
        )

    grand_mean = data[response].mean()

    treatment_totals = data.groupby(
        treatment
    )[response].sum()

    row_totals = data.groupby(
        row
    )[response].sum()

    column_totals = data.groupby(
        column
    )[response].sum()

    total_sum = data[response].sum()

    correction_factor = (
        total_sum ** 2 / (k * k)
    )

    ss_total = (
        (data[response] ** 2).sum()
        - correction_factor
    )

    ss_treatment = (
        (treatment_totals ** 2).sum() / k
        - correction_factor
    )

    ss_row = (
        (row_totals ** 2).sum() / k
        - correction_factor
    )

    ss_column = (
        (column_totals ** 2).sum() / k
        - correction_factor
    )

    ss_error = (
        ss_total
        - ss_treatment
        - ss_row
        - ss_column
    )

    df_treatment = k - 1
    df_row = k - 1
    df_column = k - 1

    df_error = (k - 1) * (k - 2)
    df_total = k * k - 1

    if df_error <= 0:
        raise ValueError(
            "درجات حرية الخطأ غير كافية للتحليل."
        )

    ms_treatment = ss_treatment / df_treatment
    ms_row = ss_row / df_row
    ms_column = ss_column / df_column
    ms_error = ss_error / df_error

    if ms_error < 0 and abs(ms_error) < 1e-10:
        ms_error = 0

    if ms_error < 0:
        raise ValueError(
            "حدث خطأ في حساب متوسط مربعات الخطأ."
        )

    def calculate_f(ms_value, df_value):

        if ms_error == 0:
            if ms_value > 0:
                return float("inf"), 0.0

            return 0.0, 1.0

        f_value = ms_value / ms_error

        p_value = f_distribution.sf(
            f_value,
            df_value,
            df_error
        )

        return f_value, p_value

    f_treatment, p_treatment = calculate_f(
        ms_treatment,
        df_treatment
    )

    f_row, p_row = calculate_f(
        ms_row,
        df_row
    )

    f_column, p_column = calculate_f(
        ms_column,
        df_column
    )

    means = data.groupby(
        treatment
    )[response].mean().to_dict()

    posthoc = calculate_posthoc(
        means,
        ms_error,
        df_error,
        k
    )

    return {
        "design": "Latin Square",
        "means": means,
        "grand_mean": grand_mean,
        "number_of_treatments": k,
        "number_of_rows": k,
        "number_of_columns": k,
        "anova": {
            "treatment": {
                "ss": ss_treatment,
                "df": df_treatment,
                "ms": ms_treatment,
                "f": f_treatment,
                "p": p_treatment
            },
            "row": {
                "ss": ss_row,
                "df": df_row,
                "ms": ms_row,
                "f": f_row,
                "p": p_row
            },
            "column": {
                "ss": ss_column,
                "df": df_column,
                "ms": ms_column,
                "f": f_column,
                "p": p_column
            },
            "error": {
                "ss": ss_error,
                "df": df_error,
                "ms": ms_error
            },
            "total": {
                "ss": ss_total,
                "df": df_total
            }
        },
        "posthoc": posthoc
    }


# ==========================================
# إعداد التقرير
# ==========================================

def format_report(result):

    lines = []

    lines.append("=" * 65)
    lines.append("تقرير التحليل الإحصائي الزراعي")
    lines.append("=" * 65)

    if result["design"] == "RCBD":
        lines.append(
            "تصميم القطاعات الكاملة العشوائية RCBD"
        )
        lines.append(
            f"عدد المعاملات: {result['number_of_treatments']}"
        )
        lines.append(
            f"عدد القطاعات: {result['number_of_blocks']}"
        )

    else:
        lines.append(
            "تصميم المربع اللاتيني Latin Square Design"
        )
        lines.append(
            f"عدد المعاملات: {result['number_of_treatments']}"
        )
        lines.append(
            f"عدد الصفوف: {result['number_of_rows']}"
        )
        lines.append(
            f"عدد الأعمدة: {result['number_of_columns']}"
        )

    lines.append(
        f"المتوسط العام: {result['grand_mean']:.4f}"
    )

    lines.append("")
    lines.append("متوسطات المعاملات")

    for name, mean in result["means"].items():
        lines.append(f"{name}: {mean:.4f}")

    lines.append("")
    lines.append("تحليل التباين ANOVA")
    lines.append(
        "مصدر التباين | SS | df | MS | F | P-value"
    )

    anova = result["anova"]

    if result["design"] == "RCBD":
        sources = [
            ("treatment", "المعاملات"),
            ("block", "القطاعات")
        ]
    else:
        sources = [
            ("row", "الصفوف"),
            ("column", "الأعمدة"),
            ("treatment", "المعاملات")
        ]

    for key, label in sources:

        item = anova[key]

        p_text = (
            f"{item['p']:.6f}"
            if item["p"] is not None else "غير متاح"
        )

        lines.append(
            f"{label} | {item['ss']:.4f} | "
            f"{item['df']} | {item['ms']:.4f} | "
            f"{item['f']:.4f} | {p_text}"
        )

    item = anova["error"]

    lines.append(
        f"الخطأ | {item['ss']:.4f} | "
        f"{item['df']} | {item['ms']:.4f}"
    )

    item = anova["total"]

    lines.append(
        f"الكلي | {item['ss']:.4f} | {item['df']}"
    )

    posthoc = result["posthoc"]

    # LSD
    lines.append("")
    lines.append("اختبار أقل فرق معنوي LSD")

    lsd = posthoc["lsd"]

    if lsd is None:
        lines.append("اختبار LSD غير متاح.")
    else:
        lines.append(
            f"LSD عند 5%: {lsd['lsd_5']:.4f}"
        )
        lines.append(
            f"LSD عند 1%: {lsd['lsd_1']:.4f}"
        )

    # Duncan
    lines.append("")
    lines.append("اختبار دنكن Duncan")

    for alpha_label, key in [
        ("5%", "duncan_5"),
        ("1%", "duncan_1")
    ]:

        lines.append("")
        lines.append(
            f"نتائج دنكن عند مستوى {alpha_label}:"
        )

        duncan = posthoc[key]

        if duncan is None:
            lines.append("اختبار دنكن غير متاح.")
            continue

        for item in duncan["comparisons"]:

            decision = (
                "معنوي"
                if item["significant"]
                else "غير معنوي"
            )

            lines.append(
                f"{item['name1']} مع {item['name2']}: "
                f"الفرق {item['difference']:.4f}، "
                f"P-value = {item['p_value']:.6f}، "
                f"{decision}"
            )

        lines.append("مجموعات الحروف:")

        for name, letter in duncan["letters"].items():
            lines.append(f"{name}: {letter}")

    # Tukey
    lines.append("")
    lines.append("اختبار توكي Tukey HSD")

    tukey = posthoc["tukey"]

    if tukey is None:
        lines.append("اختبار Tukey غير متاح.")
    else:
        lines.append(
            f"HSD عند 5%: {tukey['hsd_5']:.4f}"
        )
        lines.append(
            f"HSD عند 1%: {tukey['hsd_1']:.4f}"
        )

        for item in tukey["comparisons"]:

            decision_5 = (
                "معنوي"
                if item["significant_5"]
                else "غير معنوي"
            )

            decision_1 = (
                "معنوي"
                if item["significant_1"]
                else "غير معنوي"
            )

            lines.append(
                f"{item['name1']} مع {item['name2']}: "
                f"الفرق {item['difference']:.4f}، "
                f"P-value = {item['p_value']:.6f}، "
                f"5%: {decision_5}، "
                f"1%: {decision_1}"
            )

    # Scheffe
    lines.append("")
    lines.append("اختبار شيفيه Scheffe")

    scheffe = posthoc["scheffe"]

    if scheffe is None:
        lines.append("اختبار Scheffe غير متاح.")
    else:
        lines.append(
            f"الفرق الحرج عند 5%: {scheffe['critical_5']:.4f}"
        )
        lines.append(
            f"الفرق الحرج عند 1%: {scheffe['critical_1']:.4f}"
        )

        for item in scheffe["comparisons"]:

            decision_5 = (
                "معنوي"
                if item["significant_5"]
                else "غير معنوي"
            )

            decision_1 = (
                "معنوي"
                if item["significant_1"]
                else "غير معنوي"
            )

            lines.append(
                f"{item['name1']} مع {item['name2']}: "
                f"الفرق {item['difference']:.4f}، "
                f"P-value = {item['p_value']:.6f}، "
                f"5%: {decision_5}، "
                f"1%: {decision_1}"
            )

    lines.append("")
    lines.append("=" * 65)
    lines.append("نهاية التقرير")
    lines.append("=" * 65)

    return "\n".join(lines)

# ==========================================
# تشغيل التحليل وحفظ التقرير
# ==========================================

def save_report(report, file_name):

    desktop = os.path.join(
        os.path.expanduser("~"),
        "Desktop"
    )

    report_folder = os.path.join(
        desktop,
        "Agri Statistical Reports"
    )

    os.makedirs(report_folder, exist_ok=True)

    report_path = os.path.join(
        report_folder,
        file_name
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    print("\nتم حفظ التقرير بنجاح.")
    print("مسار التقرير:")
    print(report_path)


def run_rcbd(df):

    print("\nاختر الأعمدة المطلوبة للتحليل.")

    treatment = choose_column(df, "المعاملات")
    block = choose_column(df, "القطاعات")
    response = choose_column(df, "الصفة المدروسة")

    try:
        result = analyze_rcbd(
            df,
            treatment,
            block,
            response
        )

        report = format_report(result)

        print("\n")
        print(report)

        save_report(
            report,
            "RCBD_Report.txt"
        )

    except Exception as error:
        print("\nحدث خطأ أثناء التحليل:")
        print(error)


def run_latin_square(df):

    print("\nاختر الأعمدة المطلوبة لتحليل المربع اللاتيني.")

    treatment = choose_column(df, "المعاملات")
    row = choose_column(df, "الصفوف")
    column = choose_column(df, "الأعمدة")
    response = choose_column(df, "الصفة المدروسة")

    try:
        result = analyze_latin_square(
            df,
            treatment,
            row,
            column,
            response
        )

        report = format_report(result)

        print("\n")
        print(report)

        save_report(
            report,
            "Latin_Square_Report.txt"
        )

    except Exception as error:
        print("\nحدث خطأ أثناء التحليل:")
        print(error)


# ==========================================
# القائمة الرئيسية
# ==========================================

def main():

    df = None

    while True:

        print("\n")
        print("=" * 50)
        print("برنامج التحليل الإحصائي الزراعي")
        print("=" * 50)

        print("1 - تحميل ملف البيانات")
        print("2 - تحليل القطاعات الكاملة العشوائية RCBD")
        print("3 - تحليل المربع اللاتيني Latin Square")
        print("0 - خروج")

        choice = ask("اختر العملية المطلوبة:")

        if choice == "1":
            df = load_file()

        elif choice == "2":

            if df is None:
                print("يجب تحميل ملف البيانات أولاً.")
            else:
                run_rcbd(df)

        elif choice == "3":

            if df is None:
                print("يجب تحميل ملف البيانات أولاً.")
            else:
                run_latin_square(df)

        elif choice == "0":
            print("تم إنهاء البرنامج.")
            break

        else:
            print("اختيار غير صحيح.")


if __name__ == "__main__":
    main()

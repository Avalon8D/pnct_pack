#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector_util.c>

#define ABSPATH "/home/sianna/issues/pnct_project/"

int get_weekday (char *str)
{
    struct tm tm;
    strptime (str, "%Y-%m-%d", &tm);
    time_t t = mktime (&tm);
    return localtime (&t)->tm_wday;
}

void week_day_label (char *eq)
{
    static char date_path[512];
    sprintf (date_path, ABSPATH "PNCT_eqs/%s/lista_datas_%s.csv", eq, eq);
    FILE *eq_date = fopen (date_path, "r");

    if (!eq_date)
    {
        printf ("Equipament not fount\n");
        exit (-1);
    }

    static char wday_label_path[512];
    sprintf (wday_label_path, ABSPATH "PNCT_eqs/%s/wday_label_%s.csv", eq, eq);
    FILE *eq_label = fopen (wday_label_path, "w");

    if (!eq_label)
    {
        printf ("Unable to create label file\n");
        exit (-1);
    }

    static char curr_day_date[64];
    int day_count[7] = {};
    int curr_day_id;

    while (!feof (eq_date))
    {
        fscanf (eq_date, "%d,%s", &curr_day_id, curr_day_date);
        int wday = get_weekday (curr_day_date);
        day_count[wday]++;
        fprintf (eq_label, "%d,%d\n", curr_day_id, wday);
    }

    static char day_count_path[512];
    sprintf (day_count_path, ABSPATH "PNCT_eqs/%s/day_count_%s.csv", eq, eq);
    FILE *eq_day_count = fopen (day_count_path, "w");

    if (!eq_day_count)
    {
        printf ("Unable to create day count file\n");
        exit (-1);
    }

    fprintf (eq_day_count, "%d\n", curr_day_id + 1);

    int wday_cnt = 0;
    fprintf (eq_day_count, "%d", day_count[wday_cnt]);

    while (++wday_cnt < 7)
    { fprintf (eq_day_count, ",%d", day_count[wday_cnt]); }
}

int main (int argc, char const *argv[])
{

    FILE *eqs = fopen (ABSPATH "eq_ids", "r");

    char eq[16];

    while (!feof (eqs))
    {
        fscanf (eqs, "%s", eq);
        week_day_label (eq);
    }

    return 0;
}

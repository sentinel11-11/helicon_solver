# helicon_solver

FDFD-расчёт геликонных волн и радиальных собственных мод для ПЛМ-М.

## Быстрый запуск на ПК

```powershell
git pull
python -m pip install -r requirements.txt
copy config.yaml config_local.yaml
python run_all.py --config config_local.yaml --frequencies 5 7.5 9.8 12 15
```

Для CUDA установите CuPy и используйте `config_cuda.yaml`:

```powershell
python -m pip install cupy-cuda12x
copy config_cuda.yaml config_cuda_local.yaml
python run_all.py --config config_cuda_local.yaml --frequencies 5 7.5 9.8 12 15
```

Поле магнитной системы рассчитывается с учётом восьми магнитов длиной 0,608 м и квадратным сечением 0,018 м. Две секции соленоида задаются в `config.yaml`.

2-D FDFD использует осесимметричное усреднение дискретного поля каспов. Поле без усреднения рассчитывается отдельно:

```powershell
python magnetic_3d.py --config config.yaml
```

Параметры с отметкой `provisional` требуют замены измеренными координатами, токами, полярностями и размерами установки.

## Итерационный подбор зонда

Сначала запустите расчёт с сохранением `fields.npz`:

```powershell
python run_all.py --config config.yaml
```

После этого выполните подбор трёхкомпонентного зонда:

```powershell
python probe_optimization.py --config probe_optimization.yaml
```

Результаты появятся в `probe_optimization_results`.

Для подгонки магнитной модели по измеренному CSV-файлу используйте колонки `r_m`, `z_m`, `B_r_T`, `B_z_T` и `sigma_B_T`:

```powershell
python field_calibration.py --config config.yaml --measurements data/plmm_field.csv --output-config config_calibrated.yaml --report field_calibration_report.json
```

Параметры подгонки добавляются в секцию `field_calibration.parameters` конфигурации. Каждый элемент содержит путь YAML, начальное значение и границы `lower`, `upper`.

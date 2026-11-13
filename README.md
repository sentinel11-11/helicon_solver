# helicon_solver

FDFD-расчёт геликонных волн и радиальных собственных мод для ПЛМ-М.

## Быстрый запуск на ПК

```powershell
git pull
python -m pip install -r requirements.txt
python run_all.py --config config.yaml --frequencies 5 7.5 9.8 12 15
```

Для CUDA установите CuPy и используйте `config_cuda.yaml`:

```powershell
python -m pip install cupy-cuda12x
python run_all.py --config config_cuda.yaml --frequencies 5 7.5 9.8 12 15
```

Поле магнитной системы рассчитывается с учётом восьми магнитов длиной 0,608 м и квадратным сечением 0,018 м. Две секции соленоида задаются в `config.yaml`.

2-D FDFD использует осесимметричное усреднение дискретного поля каспов. Поле без усреднения рассчитывается отдельно:

```powershell
python magnetic_3d.py --config config.yaml
```

Параметры с отметкой `provisional` требуют замены измеренными координатами, токами, полярностями и размерами установки.

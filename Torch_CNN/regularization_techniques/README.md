# Regularization Techniques (CNN)

Bu klasör, CNN tabanlı modellerde kullanılan **regularization** tekniklerini modüler ve tekrar kullanılabilir şekilde sunar. Amaç: eğitim stabilitesi, genelleme (generalization) ve overfitting kontrolü için farklı regularization ailelerini tek bir yerde, tutarlı bir yapıyla toplamak.

---

## Klasör Yapısı

```text
regularization_techniques/
├─ dropout_based/
│  ├─ spatial_dropout/        # ❗ Zorunlu
│  ├─ drop_block/             # ❗ Zorunlu
│
├─ path_based/
│  ├─ drop_path/              # ❗ Zorunlu
│  ├─ stochastic_depth/       # ❗ Zorunlu
│
├─ mask_based/
│  ├─ cutout/                 # ❗ Zorunlu
└─ README.md
```

Not: drop_path ve stochastic_depth pratikte aynı ailenin farklı isimlendirmeleridir. Bu repo düzeninde ikisi ayrı klasörlenir çünkü kullanım biçimi (residual özel vs genel) ve literatür referansı farklılaşır.

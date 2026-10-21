### Solo OOF
| tag | OOF | hi | lo | corr BL1/Bo/Bt |
|---|---|---|---|---|
| d11_multilayer | 12.994 | -23.5 | 16.9 | 0.871/0.968/0.939 |
| d11_cropaug | 13.589 | -22.5 | 18.1 | 0.855/0.956/0.925 |

### Nested CV (10×5, NNLS+intercept) vs base
| 구성 | nCV | Δ | hi | lo | full NNLS w |
|---|---|---|---|---|---|
| base | 12.601 | 0 | -27.2 | 17.4 | [0.44, 0.283, 0.277, 0.0] |
| +d11_multilayer | 12.587 | -0.013 | -26.6 | 17.3 | [0.431, 0.047, 0.225, 0.297, 0.0] |
| +d11_cropaug | 12.607 | +0.006 | -27.2 | 17.4 | [0.44, 0.283, 0.277, 0.0, 0.0] |
| +d11_multilayer+d11_cropaug | 12.589 | -0.012 | -26.6 | 17.3 | [0.431, 0.047, 0.225, 0.297, 0.0, 0.0] |
| +d11_multilayer+d8_dldl_orig | 12.601 | +0.001 | -26.6 | 17.4 | [0.426, 0.0, 0.205, 0.243, 0.126, 0.0] |
| Borig avg ← +d11_multilayer | 12.581 | -0.020 | -27.0 | 17.4 | [0.428, 0.343, 0.229, 0.0] |
| Borig avg ← +d11_cropaug | 12.615 | +0.014 | -27.1 | 17.4 | [0.449, 0.248, 0.304, 0.0] |
| Borig avg ← +d11_multilayer+d11_cropaug | 12.598 | -0.002 | -26.9 | 17.4 | [0.437, 0.304, 0.258, 0.0] |
| Borig avg ← +d11_multilayer+d8_dldl_orig+d10_canon | 12.563 | -0.037 | -26.9 | 17.5 | [0.4, 0.412, 0.188, 0.0] |
| Borig avg ← +d11_multilayer+d11_cropaug+d8_dldl_orig+d10_canon | 12.581 | -0.020 | -26.8 | 17.5 | [0.412, 0.373, 0.215, 0.0] |
| [sel-bias] drop_s2+s3, Borig ← +d10_canon+d8_dldl_orig | 12.563 | -0.037 | -26.8 | 17.5 | [0.396, 0.364, 0.24, 0.0] |
| [sel-bias] drop_s2+s3, Borig ← +d10_canon+d8_dldl_orig+d11_multilayer | 12.552 | -0.049 | -26.7 | 17.5 | [0.392, 0.393, 0.215, 0.0] |
| [sel-bias] drop_s2+s3, Borig ← +d10_canon+d8_dldl_orig+d11_multilayer+d11_cropaug | 12.569 | -0.031 | -26.7 | 17.4 | [0.405, 0.352, 0.243, 0.0] |

BEST(all): [sel-bias] drop_s2+s3, Borig ← +d10_canon+d8_dldl_orig+d11_multilayer nCV=12.552 Δ-0.049 hi=-26.7
BEST(no post-hoc seed drop): Borig avg ← +d11_multilayer+d8_dldl_orig+d10_canon nCV=12.563 Δ-0.037 hi=-26.9
no candidate (gain < 0.05)

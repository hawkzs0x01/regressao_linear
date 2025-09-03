# Portfólio em Rust: Implementação de Regressão Linear Pura para Séries Temporais

## Cenário

A análise de séries temporais é fundamental em áreas como finanças, meteorologia, IoT, e ciência de dados, pois permite entender padrões, tendências e realizar previsões.  
Este projeto simula um desafio proposto pela startup fictícia **TimeWise Analytics**, que busca uma solução eficiente e personalizável para análises de séries temporais em Rust.  
O objetivo é criar uma biblioteca capaz de importar, analisar e prever valores em séries temporais usando **regressão linear "pura"**, sem dependências externas para os cálculos.

---

## Objetivos do Projeto

- **Implementar regressão linear pura**: Calcular os coeficientes da reta (inclinação e intercepto) que melhor ajustam uma série temporal.
- **Calcular métricas de avaliação**: Implementar funções para R² (coeficiente de determinação) e MSE (erro quadrático médio).
- **Realizar previsões**: Prever valores futuros com base na reta ajustada.
- **Cobertura de testes**: Garantir robustez e qualidade do código com testes unitários.
- **Documentação clara**: Explicar conceitos, implementação, exemplos de uso e limitações.

---

## Estrutura do Projeto

```
regressao_linear/
├── src/
│   ├── lib.rs         # Implementação da biblioteca e testes
│   └── main.rs        # Exemplo de uso básico
├── benches/
│   └── benchmarks.rs  # Benchmarks de desempenho
├── examples/
│   ├── analise_completa.rs   # Exemplo completo de análise
│   └── exemplo_basico.rs     # Exemplo básico de uso
└── README.md          # Documentação completa
```

---

## Implementação

### 1. Regressão Linear

A função principal recebe um array de números (`y`) representando a série temporal e calcula os coeficientes da reta:

- **Inclinação (a)**: Representa a taxa de variação dos dados ao longo do tempo.
- **Intercepto (b)**: Valor inicial da reta.

A implementação não utiliza crates de terceiros para cálculo estatístico.

```rust
/// Calcula a regressão linear para uma série temporal
pub fn regressao_linear(y: &[f64]) -> Resultado<(f64, f64)>
```

Para dados XY arbitrários, usamos:

```rust
pub fn regressao_linear_xy(x: &[f64], y: &[f64]) -> Resultado<(f64, f64)>
```

### 2. Métricas de Avaliação

- **Coeficiente de determinação (R²):** Mede o quanto a reta explica a variação dos dados.
- **Erro Quadrático Médio (MSE):** Mede o erro médio entre valores reais e previstos.

```rust
pub fn calcular_r2(y_real: &[f64], y_previsto: &[f64]) -> Resultado<f64>
pub fn calcular_mse(y_real: &[f64], y_previsto: &[f64]) -> Resultado<f64>
```

### 3. Previsões

Utilizando os coeficientes calculados, é possível prever valores futuros em séries temporais.

```rust
pub fn prever_valores(inicio: usize, n_valores: usize, inclinacao: f64, intercepto: f64) -> Vec<f64>
```

---

## Exemplos de Uso

### Exemplo Básico

```rust
let vendas = vec![100.0, 120.0, 140.0, 160.0, 180.0, 200.0];
let (inclinacao, intercepto) = regressao_linear(&vendas).unwrap();
let previstos = prever_valores(vendas.len(), 3, inclinacao, intercepto);
// Previsões: [220.0, 240.0, 260.0]
```

### Exemplo Completo

- Análise de séries temporais com estatísticas, regressão, métricas e previsões futuras.
- Comparação entre valores reais e previstos.
- Interpretação de tendência e qualidade do ajuste.

Veja [`examples/analise_completa.rs`](examples/analise_completa.rs) para um exemplo completo.

---

## Testes Unitários

O projeto inclui **19 testes unitários**, cobrindo:

- Série temporal perfeita
- Dados vazios/insuficientes
- XY de tamanhos diferentes
- Cálculo de R², MSE, MAE
- Previsões futuras e tendências negativas
- Casos extremos (valores grandes/pequenos)
- Estatísticas descritivas

Todos os testes passam com sucesso:

```text
test result: ok. 19 passed; 0 failed; 0 ignored; finished in 0.00s
```

---

## Benchmarks

Benchmarks medem o desempenho da biblioteca em séries de diferentes tamanhos (pequena, média, grande).  
Resultados mostram que a implementação é eficiente para uso prático.

---

## Desafios e Questões Exploratórias

1. **Principais desafios ao implementar regressão linear "pura" em Rust:**
   - Lidar com precisão numérica e limitações de ponto flutuante.
   - Garantir segurança contra entradas inválidas (vetor vazio, variância zero).
   - Manter o código enxuto e eficiente sem crates externos.

2. **Como lidar com entradas inválidas?**
   - Uso de tipos Result e enum para erros (`RegressaoError`), retornando erros claros para cada caso.

3. **Vantagens e desvantagens de Rust para algoritmos numéricos:**
   - Vantagens: Segurança de memória, alta performance, erros em tempo de compilação.
   - Desvantagens: Ecosistema menor para matemática comparado a Python/R, mais verboso para operações matemáticas.

4. **Testes unitários e qualidade:**
   - Cobrem cenários comuns e extremos, prevenindo bugs e garantindo confiabilidade.

5. **Importância da documentação clara:**
   - Facilita uso da biblioteca por outros desenvolvedores e garante entendimento dos métodos e limitações.

6. **Regressão linear para previsões em séries temporais:**
   - Permite prever valores futuros com base em tendências históricas, útil para planejamento e tomada de decisão.

7. **Limitações da regressão linear:**
   - Não captura sazonalidade, não se ajusta bem a dados não lineares, sensível a outliers.

8. **Métricas de avaliação ajudam a determinar qualidade:**
   - R² próximo de 1 indica ajuste excelente, MSE baixo indica boa precisão.

---

## Como usar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/hawkzs0x01/regressao_linear.git
   cd regressao_linear
   ```

2. **Compile e rode os exemplos:**
   ```bash
   cargo run --example exemplo_basico
   cargo run --example analise_completa
   ```

3. **Execute os testes:**
   ```bash
   cargo test
   ```

4. **Rode benchmarks (opcional):**
   ```bash
   cargo bench
   ```

---

## Limitações, Sugestões e Expansão

- **Limitações:** Não há suporte para importação de CSV/JSON, visualização gráfica ou modelos não lineares.
- **Sugestões:**  
  - Integrar parsing de arquivos (usando `serde` e `csv`).
  - Acrescentar gráficos (usando `plotters`).
  - Implementar modelos como regressão polinomial ou ARIMA.
- **Expansão:**  
  - Adicionar interface web ou CLI para uso interativo.

---



## Autor

- **Nome:** [Guilherme Rodrigues]
- **GitHub:** [hawkzs0x01](https://github.com/hawkzs0x01)

---

## Licença

Este projeto está sob a licença MIT.

---

## Referências

- Documentação oficial do Rust: [https://doc.rust-lang.org/](https://doc.rust-lang.org/)
- Artigos sobre regressão linear e séries temporais
- [TimeWise Analytics - Startup fictícia do projeto]

---

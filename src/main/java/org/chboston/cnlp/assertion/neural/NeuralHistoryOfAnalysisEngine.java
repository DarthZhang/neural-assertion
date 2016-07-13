package org.chboston.cnlp.assertion.neural;

import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;

public class NeuralHistoryOfAnalysisEngine extends
    NeuralAssertionStatusAnalysisEngine {

  @Override
  public String getLabel(IdentifiedAnnotation ent) {
    return String.valueOf(ent.getHistoryOf());
  }

  @Override
  public void applyLabel(IdentifiedAnnotation ent, String label) {
    ent.setHistoryOf(Integer.parseInt(label));
  }

}

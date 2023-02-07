"""
TOXCAST dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

TOXCAST_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"
TOXCAST_TASKS = [
    'ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive',
    'APR_HepG2_CellCycleArrest_24h_dn', 'APR_HepG2_CellCycleArrest_24h_up',
    'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn',
    'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_24h_dn',
    'APR_HepG2_MicrotubuleCSK_24h_up', 'APR_HepG2_MicrotubuleCSK_72h_dn',
    'APR_HepG2_MicrotubuleCSK_72h_up', 'APR_HepG2_MitoMass_24h_dn',
    'APR_HepG2_MitoMass_24h_up', 'APR_HepG2_MitoMass_72h_dn',
    'APR_HepG2_MitoMass_72h_up', 'APR_HepG2_MitoMembPot_1h_dn',
    'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
    'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up',
    'APR_HepG2_NuclearSize_24h_dn', 'APR_HepG2_NuclearSize_72h_dn',
    'APR_HepG2_NuclearSize_72h_up', 'APR_HepG2_OxidativeStress_24h_up',
    'APR_HepG2_OxidativeStress_72h_up', 'APR_HepG2_StressKinase_1h_up',
    'APR_HepG2_StressKinase_24h_up', 'APR_HepG2_StressKinase_72h_up',
    'APR_HepG2_p53Act_24h_up', 'APR_HepG2_p53Act_72h_up',
    'APR_Hepat_Apoptosis_24hr_up', 'APR_Hepat_Apoptosis_48hr_up',
    'APR_Hepat_CellLoss_24hr_dn', 'APR_Hepat_CellLoss_48hr_dn',
    'APR_Hepat_DNADamage_24hr_up', 'APR_Hepat_DNADamage_48hr_up',
    'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up',
    'APR_Hepat_MitoFxnI_1hr_dn', 'APR_Hepat_MitoFxnI_24hr_dn',
    'APR_Hepat_MitoFxnI_48hr_dn', 'APR_Hepat_NuclearSize_24hr_dn',
    'APR_Hepat_NuclearSize_48hr_dn', 'APR_Hepat_Steatosis_24hr_up',
    'APR_Hepat_Steatosis_48hr_up', 'ATG_AP_1_CIS_dn', 'ATG_AP_1_CIS_up',
    'ATG_AP_2_CIS_dn', 'ATG_AP_2_CIS_up', 'ATG_AR_TRANS_dn', 'ATG_AR_TRANS_up',
    'ATG_Ahr_CIS_dn', 'ATG_Ahr_CIS_up', 'ATG_BRE_CIS_dn', 'ATG_BRE_CIS_up',
    'ATG_CAR_TRANS_dn', 'ATG_CAR_TRANS_up', 'ATG_CMV_CIS_dn', 'ATG_CMV_CIS_up',
    'ATG_CRE_CIS_dn', 'ATG_CRE_CIS_up', 'ATG_C_EBP_CIS_dn', 'ATG_C_EBP_CIS_up',
    'ATG_DR4_LXR_CIS_dn', 'ATG_DR4_LXR_CIS_up', 'ATG_DR5_CIS_dn',
    'ATG_DR5_CIS_up', 'ATG_E2F_CIS_dn', 'ATG_E2F_CIS_up', 'ATG_EGR_CIS_up',
    'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up', 'ATG_ERRa_TRANS_dn',
    'ATG_ERRg_TRANS_dn', 'ATG_ERRg_TRANS_up', 'ATG_ERa_TRANS_up',
    'ATG_E_Box_CIS_dn', 'ATG_E_Box_CIS_up', 'ATG_Ets_CIS_dn', 'ATG_Ets_CIS_up',
    'ATG_FXR_TRANS_up', 'ATG_FoxA2_CIS_dn', 'ATG_FoxA2_CIS_up',
    'ATG_FoxO_CIS_dn', 'ATG_FoxO_CIS_up', 'ATG_GAL4_TRANS_dn',
    'ATG_GATA_CIS_dn', 'ATG_GATA_CIS_up', 'ATG_GLI_CIS_dn', 'ATG_GLI_CIS_up',
    'ATG_GRE_CIS_dn', 'ATG_GRE_CIS_up', 'ATG_GR_TRANS_dn', 'ATG_GR_TRANS_up',
    'ATG_HIF1a_CIS_dn', 'ATG_HIF1a_CIS_up', 'ATG_HNF4a_TRANS_dn',
    'ATG_HNF4a_TRANS_up', 'ATG_HNF6_CIS_dn', 'ATG_HNF6_CIS_up',
    'ATG_HSE_CIS_dn', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn', 'ATG_IR1_CIS_up',
    'ATG_ISRE_CIS_dn', 'ATG_ISRE_CIS_up', 'ATG_LXRa_TRANS_dn',
    'ATG_LXRa_TRANS_up', 'ATG_LXRb_TRANS_dn', 'ATG_LXRb_TRANS_up',
    'ATG_MRE_CIS_up', 'ATG_M_06_TRANS_up', 'ATG_M_19_CIS_dn',
    'ATG_M_19_TRANS_dn', 'ATG_M_19_TRANS_up', 'ATG_M_32_CIS_dn',
    'ATG_M_32_CIS_up', 'ATG_M_32_TRANS_dn', 'ATG_M_32_TRANS_up',
    'ATG_M_61_TRANS_up', 'ATG_Myb_CIS_dn', 'ATG_Myb_CIS_up', 'ATG_Myc_CIS_dn',
    'ATG_Myc_CIS_up', 'ATG_NFI_CIS_dn', 'ATG_NFI_CIS_up', 'ATG_NF_kB_CIS_dn',
    'ATG_NF_kB_CIS_up', 'ATG_NRF1_CIS_dn', 'ATG_NRF1_CIS_up',
    'ATG_NRF2_ARE_CIS_dn', 'ATG_NRF2_ARE_CIS_up', 'ATG_NURR1_TRANS_dn',
    'ATG_NURR1_TRANS_up', 'ATG_Oct_MLP_CIS_dn', 'ATG_Oct_MLP_CIS_up',
    'ATG_PBREM_CIS_dn', 'ATG_PBREM_CIS_up', 'ATG_PPARa_TRANS_dn',
    'ATG_PPARa_TRANS_up', 'ATG_PPARd_TRANS_up', 'ATG_PPARg_TRANS_up',
    'ATG_PPRE_CIS_dn', 'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up',
    'ATG_PXR_TRANS_dn', 'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up',
    'ATG_RARa_TRANS_dn', 'ATG_RARa_TRANS_up', 'ATG_RARb_TRANS_dn',
    'ATG_RARb_TRANS_up', 'ATG_RARg_TRANS_dn', 'ATG_RARg_TRANS_up',
    'ATG_RORE_CIS_dn', 'ATG_RORE_CIS_up', 'ATG_RORb_TRANS_dn',
    'ATG_RORg_TRANS_dn', 'ATG_RORg_TRANS_up', 'ATG_RXRa_TRANS_dn',
    'ATG_RXRa_TRANS_up', 'ATG_RXRb_TRANS_dn', 'ATG_RXRb_TRANS_up',
    'ATG_SREBP_CIS_dn', 'ATG_SREBP_CIS_up', 'ATG_STAT3_CIS_dn',
    'ATG_STAT3_CIS_up', 'ATG_Sox_CIS_dn', 'ATG_Sox_CIS_up', 'ATG_Sp1_CIS_dn',
    'ATG_Sp1_CIS_up', 'ATG_TAL_CIS_dn', 'ATG_TAL_CIS_up', 'ATG_TA_CIS_dn',
    'ATG_TA_CIS_up', 'ATG_TCF_b_cat_CIS_dn', 'ATG_TCF_b_cat_CIS_up',
    'ATG_TGFb_CIS_dn', 'ATG_TGFb_CIS_up', 'ATG_THRa1_TRANS_dn',
    'ATG_THRa1_TRANS_up', 'ATG_VDRE_CIS_dn', 'ATG_VDRE_CIS_up',
    'ATG_VDR_TRANS_dn', 'ATG_VDR_TRANS_up', 'ATG_XTT_Cytotoxicity_up',
    'ATG_Xbp1_CIS_dn', 'ATG_Xbp1_CIS_up', 'ATG_p53_CIS_dn', 'ATG_p53_CIS_up',
    'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down', 'BSK_3C_ICAM1_down',
    'BSK_3C_IL8_down', 'BSK_3C_MCP1_down', 'BSK_3C_MIG_down',
    'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down',
    'BSK_3C_Thrombomodulin_down', 'BSK_3C_Thrombomodulin_up',
    'BSK_3C_TissueFactor_down', 'BSK_3C_TissueFactor_up', 'BSK_3C_VCAM1_down',
    'BSK_3C_Vis_down', 'BSK_3C_uPAR_down', 'BSK_4H_Eotaxin3_down',
    'BSK_4H_MCP1_down', 'BSK_4H_Pselectin_down', 'BSK_4H_Pselectin_up',
    'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down', 'BSK_4H_VEGFRII_down',
    'BSK_4H_uPAR_down', 'BSK_4H_uPAR_up', 'BSK_BE3C_HLADR_down',
    'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down',
    'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down',
    'BSK_BE3C_SRB_down', 'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down',
    'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPAR_up', 'BSK_BE3C_uPA_down',
    'BSK_CASM3C_HLADR_down', 'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL6_up',
    'BSK_CASM3C_IL8_down', 'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_LDLR_up',
    'BSK_CASM3C_MCP1_down', 'BSK_CASM3C_MCP1_up', 'BSK_CASM3C_MCSF_down',
    'BSK_CASM3C_MCSF_up', 'BSK_CASM3C_MIG_down',
    'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_Proliferation_up',
    'BSK_CASM3C_SAA_down', 'BSK_CASM3C_SAA_up', 'BSK_CASM3C_SRB_down',
    'BSK_CASM3C_Thrombomodulin_down', 'BSK_CASM3C_Thrombomodulin_up',
    'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down',
    'BSK_CASM3C_VCAM1_up', 'BSK_CASM3C_uPAR_down', 'BSK_CASM3C_uPAR_up',
    'BSK_KF3CT_ICAM1_down', 'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down',
    'BSK_KF3CT_IP10_up', 'BSK_KF3CT_MCP1_down', 'BSK_KF3CT_MCP1_up',
    'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down', 'BSK_KF3CT_TGFb1_down',
    'BSK_KF3CT_TIMP2_down', 'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down',
    'BSK_LPS_Eselectin_down', 'BSK_LPS_Eselectin_up', 'BSK_LPS_IL1a_down',
    'BSK_LPS_IL1a_up', 'BSK_LPS_IL8_down', 'BSK_LPS_IL8_up',
    'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down',
    'BSK_LPS_PGE2_up', 'BSK_LPS_SRB_down', 'BSK_LPS_TNFa_down',
    'BSK_LPS_TNFa_up', 'BSK_LPS_TissueFactor_down', 'BSK_LPS_TissueFactor_up',
    'BSK_LPS_VCAM1_down', 'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down',
    'BSK_SAg_CD69_down', 'BSK_SAg_Eselectin_down', 'BSK_SAg_Eselectin_up',
    'BSK_SAg_IL8_down', 'BSK_SAg_IL8_up', 'BSK_SAg_MCP1_down',
    'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down',
    'BSK_SAg_PBMCCytotoxicity_up', 'BSK_SAg_Proliferation_down',
    'BSK_SAg_SRB_down', 'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_EGFR_down',
    'BSK_hDFCGF_EGFR_up', 'BSK_hDFCGF_IL8_down', 'BSK_hDFCGF_IP10_down',
    'BSK_hDFCGF_MCSF_down', 'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down',
    'BSK_hDFCGF_MMP1_up', 'BSK_hDFCGF_PAI1_down',
    'BSK_hDFCGF_Proliferation_down', 'BSK_hDFCGF_SRB_down',
    'BSK_hDFCGF_TIMP1_down', 'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn',
    'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn', 'CEETOX_H295R_DOC_dn',
    'CEETOX_H295R_DOC_up', 'CEETOX_H295R_ESTRADIOL_dn',
    'CEETOX_H295R_ESTRADIOL_up', 'CEETOX_H295R_ESTRONE_dn',
    'CEETOX_H295R_ESTRONE_up', 'CEETOX_H295R_OHPREG_up',
    'CEETOX_H295R_OHPROG_dn', 'CEETOX_H295R_OHPROG_up', 'CEETOX_H295R_PROG_up',
    'CEETOX_H295R_TESTO_dn', 'CLD_ABCB1_48hr', 'CLD_ABCG2_48hr',
    'CLD_CYP1A1_24hr', 'CLD_CYP1A1_48hr', 'CLD_CYP1A1_6hr', 'CLD_CYP1A2_24hr',
    'CLD_CYP1A2_48hr', 'CLD_CYP1A2_6hr', 'CLD_CYP2B6_24hr', 'CLD_CYP2B6_48hr',
    'CLD_CYP2B6_6hr', 'CLD_CYP3A4_24hr', 'CLD_CYP3A4_48hr', 'CLD_CYP3A4_6hr',
    'CLD_GSTA2_48hr', 'CLD_SULT2A_24hr', 'CLD_SULT2A_48hr', 'CLD_UGT1A1_24hr',
    'CLD_UGT1A1_48hr', 'NCCT_HEK293T_CellTiterGLO', 'NCCT_QuantiLum_inhib_2_dn',
    'NCCT_QuantiLum_inhib_dn', 'NCCT_TPO_AUR_dn', 'NCCT_TPO_GUA_dn',
    'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_ADME_hCYP19A1', 'NVS_ADME_hCYP1A1',
    'NVS_ADME_hCYP1A2', 'NVS_ADME_hCYP2A6', 'NVS_ADME_hCYP2B6',
    'NVS_ADME_hCYP2C19', 'NVS_ADME_hCYP2C9', 'NVS_ADME_hCYP2D6',
    'NVS_ADME_hCYP3A4', 'NVS_ADME_hCYP4F12', 'NVS_ADME_rCYP2C12',
    'NVS_ENZ_hAChE', 'NVS_ENZ_hAMPKa1', 'NVS_ENZ_hAurA', 'NVS_ENZ_hBACE',
    'NVS_ENZ_hCASP5', 'NVS_ENZ_hCK1D', 'NVS_ENZ_hDUSP3', 'NVS_ENZ_hES',
    'NVS_ENZ_hElastase', 'NVS_ENZ_hFGFR1', 'NVS_ENZ_hGSK3b', 'NVS_ENZ_hMMP1',
    'NVS_ENZ_hMMP13', 'NVS_ENZ_hMMP2', 'NVS_ENZ_hMMP3', 'NVS_ENZ_hMMP7',
    'NVS_ENZ_hMMP9', 'NVS_ENZ_hPDE10', 'NVS_ENZ_hPDE4A1', 'NVS_ENZ_hPDE5',
    'NVS_ENZ_hPI3Ka', 'NVS_ENZ_hPTEN', 'NVS_ENZ_hPTPN11', 'NVS_ENZ_hPTPN12',
    'NVS_ENZ_hPTPN13', 'NVS_ENZ_hPTPN9', 'NVS_ENZ_hPTPRC', 'NVS_ENZ_hSIRT1',
    'NVS_ENZ_hSIRT2', 'NVS_ENZ_hTrkA', 'NVS_ENZ_hVEGFR2', 'NVS_ENZ_oCOX1',
    'NVS_ENZ_oCOX2', 'NVS_ENZ_rAChE', 'NVS_ENZ_rCNOS', 'NVS_ENZ_rMAOAC',
    'NVS_ENZ_rMAOAP', 'NVS_ENZ_rMAOBC', 'NVS_ENZ_rMAOBP', 'NVS_ENZ_rabI2C',
    'NVS_GPCR_bAdoR_NonSelective', 'NVS_GPCR_bDR_NonSelective',
    'NVS_GPCR_g5HT4', 'NVS_GPCR_gH2', 'NVS_GPCR_gLTB4', 'NVS_GPCR_gLTD4',
    'NVS_GPCR_gMPeripheral_NonSelective', 'NVS_GPCR_gOpiateK',
    'NVS_GPCR_h5HT2A', 'NVS_GPCR_h5HT5A', 'NVS_GPCR_h5HT6', 'NVS_GPCR_h5HT7',
    'NVS_GPCR_hAT1', 'NVS_GPCR_hAdoRA1', 'NVS_GPCR_hAdoRA2a',
    'NVS_GPCR_hAdra2A', 'NVS_GPCR_hAdra2C', 'NVS_GPCR_hAdrb1',
    'NVS_GPCR_hAdrb2', 'NVS_GPCR_hAdrb3', 'NVS_GPCR_hDRD1', 'NVS_GPCR_hDRD2s',
    'NVS_GPCR_hDRD4.4', 'NVS_GPCR_hH1', 'NVS_GPCR_hLTB4_BLT1', 'NVS_GPCR_hM1',
    'NVS_GPCR_hM2', 'NVS_GPCR_hM3', 'NVS_GPCR_hM4', 'NVS_GPCR_hNK2',
    'NVS_GPCR_hOpiate_D1', 'NVS_GPCR_hOpiate_mu', 'NVS_GPCR_hTXA2',
    'NVS_GPCR_p5HT2C', 'NVS_GPCR_r5HT1_NonSelective',
    'NVS_GPCR_r5HT_NonSelective', 'NVS_GPCR_rAdra1B',
    'NVS_GPCR_rAdra1_NonSelective', 'NVS_GPCR_rAdra2_NonSelective',
    'NVS_GPCR_rAdrb_NonSelective', 'NVS_GPCR_rNK1', 'NVS_GPCR_rNK3',
    'NVS_GPCR_rOpiate_NonSelective', 'NVS_GPCR_rOpiate_NonSelectiveNa',
    'NVS_GPCR_rSST', 'NVS_GPCR_rTRH', 'NVS_GPCR_rV1', 'NVS_GPCR_rabPAF',
    'NVS_GPCR_rmAdra2B', 'NVS_IC_hKhERGCh', 'NVS_IC_rCaBTZCHL',
    'NVS_IC_rCaDHPRCh_L', 'NVS_IC_rNaCh_site2', 'NVS_LGIC_bGABARa1',
    'NVS_LGIC_h5HT3', 'NVS_LGIC_hNNR_NBungSens', 'NVS_LGIC_rGABAR_NonSelective',
    'NVS_LGIC_rNNR_BungSens', 'NVS_MP_hPBR', 'NVS_MP_rPBR', 'NVS_NR_bER',
    'NVS_NR_bPR', 'NVS_NR_cAR', 'NVS_NR_hAR', 'NVS_NR_hCAR_Antagonist',
    'NVS_NR_hER', 'NVS_NR_hFXR_Agonist', 'NVS_NR_hFXR_Antagonist', 'NVS_NR_hGR',
    'NVS_NR_hPPARa', 'NVS_NR_hPPARg', 'NVS_NR_hPR', 'NVS_NR_hPXR',
    'NVS_NR_hRAR_Antagonist', 'NVS_NR_hRARa_Agonist', 'NVS_NR_hTRa_Antagonist',
    'NVS_NR_mERa', 'NVS_NR_rAR', 'NVS_NR_rMR', 'NVS_OR_gSIGMA_NonSelective',
    'NVS_TR_gDAT', 'NVS_TR_hAdoT', 'NVS_TR_hDAT', 'NVS_TR_hNET', 'NVS_TR_hSERT',
    'NVS_TR_rNET', 'NVS_TR_rSERT', 'NVS_TR_rVMAT2', 'OT_AR_ARELUC_AG_1440',
    'OT_AR_ARSRC1_0480', 'OT_AR_ARSRC1_0960', 'OT_ER_ERaERa_0480',
    'OT_ER_ERaERa_1440', 'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440',
    'OT_ER_ERbERb_0480', 'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120',
    'OT_ERa_EREGFP_0480', 'OT_FXR_FXRSRC1_0480', 'OT_FXR_FXRSRC1_1440',
    'OT_NURR1_NURR1RXRa_0480', 'OT_NURR1_NURR1RXRa_1440',
    'TOX21_ARE_BLA_Agonist_ch1', 'TOX21_ARE_BLA_Agonist_ch2',
    'TOX21_ARE_BLA_agonist_ratio', 'TOX21_ARE_BLA_agonist_viability',
    'TOX21_AR_BLA_Agonist_ch1', 'TOX21_AR_BLA_Agonist_ch2',
    'TOX21_AR_BLA_Agonist_ratio', 'TOX21_AR_BLA_Antagonist_ch1',
    'TOX21_AR_BLA_Antagonist_ch2', 'TOX21_AR_BLA_Antagonist_ratio',
    'TOX21_AR_BLA_Antagonist_viability', 'TOX21_AR_LUC_MDAKB2_Agonist',
    'TOX21_AR_LUC_MDAKB2_Antagonist', 'TOX21_AR_LUC_MDAKB2_Antagonist2',
    'TOX21_AhR_LUC_Agonist', 'TOX21_Aromatase_Inhibition',
    'TOX21_AutoFluor_HEK293_Cell_blue', 'TOX21_AutoFluor_HEK293_Media_blue',
    'TOX21_AutoFluor_HEPG2_Cell_blue', 'TOX21_AutoFluor_HEPG2_Cell_green',
    'TOX21_AutoFluor_HEPG2_Media_blue', 'TOX21_AutoFluor_HEPG2_Media_green',
    'TOX21_ELG1_LUC_Agonist', 'TOX21_ERa_BLA_Agonist_ch1',
    'TOX21_ERa_BLA_Agonist_ch2', 'TOX21_ERa_BLA_Agonist_ratio',
    'TOX21_ERa_BLA_Antagonist_ch1', 'TOX21_ERa_BLA_Antagonist_ch2',
    'TOX21_ERa_BLA_Antagonist_ratio', 'TOX21_ERa_BLA_Antagonist_viability',
    'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_ERa_LUC_BG1_Antagonist',
    'TOX21_ESRE_BLA_ch1', 'TOX21_ESRE_BLA_ch2', 'TOX21_ESRE_BLA_ratio',
    'TOX21_ESRE_BLA_viability', 'TOX21_FXR_BLA_Antagonist_ch1',
    'TOX21_FXR_BLA_Antagonist_ch2', 'TOX21_FXR_BLA_agonist_ch2',
    'TOX21_FXR_BLA_agonist_ratio', 'TOX21_FXR_BLA_antagonist_ratio',
    'TOX21_FXR_BLA_antagonist_viability', 'TOX21_GR_BLA_Agonist_ch1',
    'TOX21_GR_BLA_Agonist_ch2', 'TOX21_GR_BLA_Agonist_ratio',
    'TOX21_GR_BLA_Antagonist_ch2', 'TOX21_GR_BLA_Antagonist_ratio',
    'TOX21_GR_BLA_Antagonist_viability', 'TOX21_HSE_BLA_agonist_ch1',
    'TOX21_HSE_BLA_agonist_ch2', 'TOX21_HSE_BLA_agonist_ratio',
    'TOX21_HSE_BLA_agonist_viability', 'TOX21_MMP_ratio_down',
    'TOX21_MMP_ratio_up', 'TOX21_MMP_viability', 'TOX21_NFkB_BLA_agonist_ch1',
    'TOX21_NFkB_BLA_agonist_ch2', 'TOX21_NFkB_BLA_agonist_ratio',
    'TOX21_NFkB_BLA_agonist_viability', 'TOX21_PPARd_BLA_Agonist_viability',
    'TOX21_PPARd_BLA_Antagonist_ch1', 'TOX21_PPARd_BLA_agonist_ch1',
    'TOX21_PPARd_BLA_agonist_ch2', 'TOX21_PPARd_BLA_agonist_ratio',
    'TOX21_PPARd_BLA_antagonist_ratio', 'TOX21_PPARd_BLA_antagonist_viability',
    'TOX21_PPARg_BLA_Agonist_ch1', 'TOX21_PPARg_BLA_Agonist_ch2',
    'TOX21_PPARg_BLA_Agonist_ratio', 'TOX21_PPARg_BLA_Antagonist_ch1',
    'TOX21_PPARg_BLA_antagonist_ratio', 'TOX21_PPARg_BLA_antagonist_viability',
    'TOX21_TR_LUC_GH3_Agonist', 'TOX21_TR_LUC_GH3_Antagonist',
    'TOX21_VDR_BLA_Agonist_viability', 'TOX21_VDR_BLA_Antagonist_ch1',
    'TOX21_VDR_BLA_agonist_ch2', 'TOX21_VDR_BLA_agonist_ratio',
    'TOX21_VDR_BLA_antagonist_ratio', 'TOX21_VDR_BLA_antagonist_viability',
    'TOX21_p53_BLA_p1_ch1', 'TOX21_p53_BLA_p1_ch2', 'TOX21_p53_BLA_p1_ratio',
    'TOX21_p53_BLA_p1_viability', 'TOX21_p53_BLA_p2_ch1',
    'TOX21_p53_BLA_p2_ch2', 'TOX21_p53_BLA_p2_ratio',
    'TOX21_p53_BLA_p2_viability', 'TOX21_p53_BLA_p3_ch1',
    'TOX21_p53_BLA_p3_ch2', 'TOX21_p53_BLA_p3_ratio',
    'TOX21_p53_BLA_p3_viability', 'TOX21_p53_BLA_p4_ch1',
    'TOX21_p53_BLA_p4_ch2', 'TOX21_p53_BLA_p4_ratio',
    'TOX21_p53_BLA_p4_viability', 'TOX21_p53_BLA_p5_ch1',
    'TOX21_p53_BLA_p5_ch2', 'TOX21_p53_BLA_p5_ratio',
    'TOX21_p53_BLA_p5_viability', 'Tanguay_ZF_120hpf_AXIS_up',
    'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_BRAI_up',
    'Tanguay_ZF_120hpf_CFIN_up', 'Tanguay_ZF_120hpf_CIRC_up',
    'Tanguay_ZF_120hpf_EYE_up', 'Tanguay_ZF_120hpf_JAW_up',
    'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_OTIC_up',
    'Tanguay_ZF_120hpf_PE_up', 'Tanguay_ZF_120hpf_PFIN_up',
    'Tanguay_ZF_120hpf_PIG_up', 'Tanguay_ZF_120hpf_SNOU_up',
    'Tanguay_ZF_120hpf_SOMI_up', 'Tanguay_ZF_120hpf_SWIM_up',
    'Tanguay_ZF_120hpf_TRUN_up', 'Tanguay_ZF_120hpf_TR_up',
    'Tanguay_ZF_120hpf_YSE_up'
]


class _ToxcastLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "toxcast_data.csv.gz")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=TOXCAST_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_toxcast(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load Toxcast dataset

    ToxCast is an extended data collection from the same
    initiative as Tox21, providing toxicology data for a large
    library of compounds based on in vitro high-throughput
    screening. The processed collection includes qualitative
    results of over 600 experiments on 8k compounds.

    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles": SMILES representation of the molecular structure
    - "ACEA_T47D_80hr_Negative" ~ "Tanguay_ZF_120hpf_YSE_up": Bioassays results.
        Please refer to the section "high-throughput assay information" at
        https://www.epa.gov/chemical-research/toxicity-forecaster-toxcasttm-data
        for details.

    Parameters
    ----------
    featurizer: Featurizer or str
        the featurizer to use for processing the data.  Alternatively you can pass
        one of the names from dc.molnet.featurizers as a shortcut.
    splitter: Splitter or str
        the splitter to use for splitting the data into training, validation, and
        test sets.  Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut.  If this is None, all the data
        will be included in a single dataset.
    transformers: list of TransformerGenerators or strings
        the Transformers to apply to the data.  Each one is specified by a
        TransformerGenerator or, as a shortcut, one of the names from
        dc.molnet.transformers.
    reload: bool
        if True, the first call for a particular featurizer and splitter will cache
        the datasets to disk, and subsequent calls will reload the cached datasets.
    data_dir: str
        a directory to save the raw data in
    save_dir: str
        a directory to save the dataset in

    References
    ----------
    .. [1] Richard, Ann M., et al. "ToxCast chemical landscape: paving the road
        to 21st century toxicology." Chemical research in toxicology 29.8 (2016):
        1225-1251.
    """
    loader = _ToxcastLoader(featurizer, splitter, transformers, TOXCAST_TASKS,
                            data_dir, save_dir, **kwargs)
    return loader.load_dataset('toxcast', reload)

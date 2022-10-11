import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as s
import shap
from econml.grf import CausalForest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# PLEとOCSの合計点をまとめたデータフレーム
# OCSは全回答あり、PLEは1期に「Yes, definitely」が1項目もなかった人
# 1844人


df = pd.read_table("TTC2022_PLE_sum.csv", delimiter=",")
print(df.head())

# 特徴量 X、アウトカム y、割り当て変数 T
Y = df['PLE_sum']

# 他のPLE項目でどうなるか
# y = df['CD65_1']

T = df['OCS_0or1']  # 強迫5点以上をtreatmentとする
X = pd.read_table("TTC2022_X_dummy.csv", delimiter=",")
X = X.drop(["Unnamed: 0"], axis=1)
print("X:\n", X.head(10))

print("1期の指標からPLEとOCSを除いたX:\n", X.head())

# SHAP
# 参考：https://github.com/microsoft/EconML/blob/main/notebooks/Generalized%20Random%20Forests.ipynb
# MicroSoftのEconML


est = CausalForest(criterion='het', n_estimators=400, min_samples_leaf=5, max_depth=None,
                   min_var_fraction_leaf=None, min_var_leaf_on_val=True,
                   min_impurity_decrease=0.0, max_samples=0.45, min_balancedness_tol=.45,
                   warm_start=False, inference=True, fit_intercept=True, subforest_size=4,
                   honest=True, verbose=0, n_jobs=-1, random_state=1235)


# est = CausalForest()
est.fit(X, T, Y)

explainer = shap.Explainer(est, shap.maskers.Independent(X, max_samples=100))
shap_values = explainer(X, check_additivity=False)

# shap.plots.bar(shap_values)
# shap.plots.beeswarm(shap_values)

# j = X.columns.get_loc("AB58")  # カラム数を抽出
# print("ランダム化前のshap value\n", shap_values[:, "AB58"].abs.mean(0).values)


Y_train, Y_val, T_train, T_val, X_train, X_val = train_test_split(Y, T, X, test_size=.2)


# ハイパーパラメータをチューニング
search_params = {
    'n_estimators': [100, 1000, 1500],
    'max_features': [i for i in range(1, X_train.shape[1])],
    'min_samples_split': [5, 20, 40],
    'max_depth': [30, 40, 60],
    'n_jobs': [1],
    'random_state': [42],
}

gsr = GridSearchCV(
    RandomForestRegressor(),
    search_params,
    cv=3,
    n_jobs=-1,
    verbose=True
)

gsr.fit(X_train, Y_train)

# 最もよかったモデル
print(gsr.best_estimator_)
print("最もよかったモデルの評価", gsr.best_estimator_.score(X_val, Y_val))

'''
# 検証
# yをランダム化
ls = []
for i in range(1000):
    Y = Y.sample(frac=1, random_state=i)

    """
    est = CausalForest(criterion='het', n_estimators=400, min_samples_leaf=5, max_depth=None,
                       min_var_fraction_leaf=None, min_var_leaf_on_val=True,
                       min_impurity_decrease=0.0, max_samples=0.45, min_balancedness_tol=.45,
                       warm_start=False, inference=True, fit_intercept=True, subforest_size=4,
                       honest=True, verbose=0, n_jobs=-1, random_state=1235)
    """

    est.fit(X, T, Y)

    explainer = shap.Explainer(est, shap.maskers.Independent(X, max_samples=100))
    shap_values = explainer(X[:200], check_additivity=False)

    print(i+1, "回目のランダム化：shap value\n", shap_values[:, "AB58"].abs.mean(0).values)
    ls.append(shap_values[:, "AB58"].abs.mean(0).values)

print(ls)
'''


'''
# 描画して検証
ls = [0.008103625161177843, 0.01117172366667255, 0.0046775375702078235, 0.006183673039127962, 0.012112344217357896, 0.013583349532909097, 0.006584343748868924, 0.022666065050558428, 0.005841297211173242, 0.009287649766806407, 0.006198635589183142, 0.007582052593908338, 0.007137272892798865, 0.009711815101520553, 0.007439703175889736, 0.009488079101602124, 0.024106990380894602, 0.0049413078626221255, 0.010038297629390583, 0.02081805176666894, 0.01349459483677864, 0.00925095418043129, 0.008875848358111399, 0.03474788836818334, 0.009856623796384701, 0.006777269718553362, 0.010868333509698548, 0.007424280665197784, 0.01311322730783504, 0.0042427274040623165, 0.013749974957735502, 0.008472436942550484, 0.007328795295125746, 0.0031201346874084267, 0.012378823882684265, 0.03358314293232979, 0.005861622953973347, 0.009730335082800957, 0.0052338925138850755, 0.02168692352379976, 0.009521116841990442, 0.009849001023488563, 0.035655826323484234, 0.0032487374977325082, 0.005253097010583951, 0.004604360855062078, 0.016376754814436206, 0.01626737359337221, 0.009980930264553444, 0.009208013624294473, 0.021346210882093146, 0.026984460623996526, 0.0045981287552882355, 0.009030632829972091, 0.005313277910940997, 0.011359621838463863, 0.007335120111460674, 0.0075219097455741565, 0.005197111846049575, 0.009291538457514617, 0.0035285602984382425, 0.004323431730876654, 0.008466269590789353, 0.009995921339737104, 0.013643014930431675, 0.006984858832009013, 0.02434964416705402, 0.0051729674976120806, 0.03809598611505208, 0.009598726589971556, 0.022820563440191472, 0.006866377352459131, 0.004826285953374281, 0.011871862284907, 0.005011423460174223, 0.017384852977304945, 0.006386133260077622, 0.009396486041484969, 0.004857239514176399, 0.007754880171125478, 0.003967722078262704, 0.004591978894648855, 0.030626344281042838, 0.0040887941129039975, 0.012625268142052663, 0.0197755488477349, 0.016521610419104903, 0.009648216490385721, 0.005156043319767833, 0.019653958409212646, 0.01717363400660834, 0.005320263174487991, 0.03747178868814453, 0.025247834104720458, 0.0031481708264102055, 0.019677435030687772, 0.01270777796446473, 0.0220988528558094, 0.012254057159781223, 0.015003721880177184, 0.01372895657461404, 0.007960234631653839, 0.013893386991421175, 0.05043030685267004, 0.005742483485045022, 0.018303939434969108, 0.008057113498761101, 0.012945040873205926, 0.004439983994560316, 0.006259150709111419, 0.01586071277906085, 0.007819626269704531, 0.014461364998912542, 0.011076289215360156, 0.02344318181264707, 0.004723874481201164, 0.011632174687253248, 0.017402007093189238, 0.020430651848718392, 0.01641293800999165, 0.003036221566272434, 0.020550969654435744, 0.02061063239004111, 0.009048265760862706, 0.009369521498196627, 0.004075674786713534, 0.01155370867231086, 0.017707165488583452, 0.009082112463070552, 0.002748260598004708, 0.004106528582560713, 0.011298246946545167, 0.008247689537108454, 0.01260174576846766, 0.007343751097624409, 0.02000285113074515, 0.010716229461421607, 0.004077483073211351, 0.0029805903373959154, 0.0038595751221202133, 0.0051671767392711, 0.02029317949860074, 0.010063240838751517, 0.0039155032654231035, 0.01416741873702872, 0.008231951847456547, 0.011483331500963322, 0.017308346074269322, 0.0036186865931901136, 0.008836661873850335, 0.007330507564807976, 0.003992097862444825, 0.007926707377332786, 0.006136552189255871, 0.0032596236366323867, 0.005021897598158285, 0.009302337512059602, 0.0016821425245774663, 0.01153591358126414, 0.008969594936151952, 0.02000060849700594, 0.00601612630269592, 0.015040879542999756, 0.011911266213737691, 0.005615053744622491, 0.003991246870649956, 0.012630665003765853, 0.006954181609252554, 0.006524595531412114, 0.004921019278990753, 0.012351267056301187, 0.00726385062211848, 0.01701797704961673, 0.009230616137776088, 0.010532577374984248, 0.002209395568596665, 0.022801931832436093, 0.005279910091315652, 0.005837049266283794, 0.007910414451378847, 0.009728607096639827, 0.011157735687374952, 0.006131615721864363, 0.005015588084923911, 0.027518103680388234, 0.005861012535050758, 0.005810099822040592, 0.011155895714680173, 0.01088374384660201, 0.002081405204095427, 0.008361613089559159, 0.05006506320112744, 0.012542897007309149, 0.01067665445952398, 0.01125659467453588, 0.002247057784194476, 0.022383121219947677, 0.00477421152915449, 0.009950749806106978, 0.008108127460626566, 0.030177989102957326, 0.008965308312275919, 0.011677531414132136, 0.0073216633128421575, 0.0029374130947880527, 0.0074214953007503935, 0.005567806274547365, 0.00855489256628789, 0.0165847132339748, 0.036243075304827475, 0.003771480799173878, 0.006687014330297825, 0.00579364040450646, 0.007930827662608317, 0.002605961775086325, 0.009178594092720413, 0.005162648288936362, 0.01151810333198996, 0.018668378452126124, 0.004582268923912306, 0.023594558563215108, 0.007017734086135897, 0.006633338150758573, 0.005427378165716254, 0.003753125029115836, 0.0215417710036043, 0.003962136731769534, 0.020807169884966604, 0.002124115266060471, 0.00438948779168204, 0.002332837131274573, 0.004708352966696838, 0.00898260351555332, 0.006904195496589818, 0.011080690290297209, 0.008812127668183529, 0.003889040731585738, 0.0021136582420753255, 0.003662314268606133, 0.011590136942753088, 0.009251341323553198, 0.006983744719838432, 0.0052415869356711485, 0.003485607096785679, 0.06877400160499254, 0.019817015336495842, 0.004706279520423778, 0.01215545621047204, 0.014925351416482045, 0.017824155419045563, 0.010019837010554693, 0.00730288301418659, 0.0034110181140320495, 0.004284385266142635, 0.008853601528379796, 0.006615121797729443, 0.011701314946452475, 0.004519298110786166, 0.01434010852351057, 0.003722001355841667, 0.004191953051486598, 0.0028433751823915376, 0.016486755134095074, 0.006195949678944454, 0.008879707813062488, 0.008159446491658628, 0.017860557732039886, 0.010451435394735016, 0.0019092626019607906, 0.007371036406460485, 0.004436029019799753, 0.011261075688325219, 0.011629244855266735, 0.014725802296292384, 0.02011108073127657, 0.02063919180212679, 0.010654195032836471, 0.0028426098055758533, 0.026304152518299633, 0.008268868216637566, 0.007873858240802474, 0.009260612357719595, 0.0203176752227062, 0.006715170686819693, 0.010087985800939716, 0.010710582410948586, 0.008581361053311412, 0.03038342274407005, 0.006378004796571985, 0.0049707699819518895, 0.00447314863591655, 0.00935820687965752, 0.013760665475396307, 0.005081250158946022, 0.003245523412869625, 0.0092450163865753, 0.007009559656177408, 0.007168118326871991, 0.013079029901370087, 0.014304594170076235, 0.05530673646592559, 0.009259803238589666, 0.008087665952749921, 0.009609698040980219, 0.008962829398472604, 0.005374172478808032, 0.004387286793045495, 0.0045725580182973315, 0.011874162802484899, 0.005499418457098364, 0.04033152219161748, 0.00796016999968906, 0.008196113703167428, 0.0033013482316677258, 0.00787403782388269, 0.0029003529781664837, 0.011579175653610945, 0.0185421997877198, 0.002981799890845014, 0.009094992418616948, 0.014889692106602159, 0.01132138119771771, 0.010827753778163971, 0.00971844655745972, 0.020346337326390494, 0.006753582467462138, 0.005213722299006622, 0.011447182160823285, 0.015823514879994036, 0.008498269073073515, 0.002092581316811265, 0.007930585018366765, 0.009706727938685072, 0.017642004341514257, 0.007581152789926546, 0.004669339354304247, 0.0025799301748869767, 0.008011090021784049, 0.009882350253765253, 0.0063729411751570295, 0.015592799549478151, 0.007057325415699961, 0.0018823646782162543, 0.011154059514487563, 0.041995108456529484, 0.01606560482924251, 0.00961435732838727, 0.010466848578527333, 0.004275367752839611, 0.016236412139044115, 0.007410092729580402, 0.004079377601279021, 0.006779249169763352, 0.007319783183840809, 0.015947130661580742, 0.004968305909144693, 0.0021484773853279876, 0.009905248648538872, 0.005418260113254383, 0.00839158709934527, 0.008368620742287385, 0.013788267011382867, 0.005244691254585631, 0.005935935321852595, 0.010883393192950461, 0.00620144345347726, 0.01137684694519994, 0.011100035668164128, 0.010550594992500918, 0.005724451518184863, 0.008923399172252539, 0.01655420159657042, 0.009945147305662068, 0.011147282848376926, 0.017703215086243642, 0.017715849512388738, 0.017187261046429148, 0.008621131649777907, 0.011430961247777306, 0.005079000807175362, 0.026416975494444474, 0.006010514679701009, 0.03850315446490781, 0.002187284333874413, 0.008484671152018291, 0.008301427722439438, 0.005569354061004493, 0.008906014115684957, 0.0038255748646337452, 0.009606270869143169, 0.00933686961189669, 0.03143698081655357, 0.006895939207562036, 0.00509396124495106, 0.01253505593908194, 0.014986989932903816, 0.002302169084912748, 0.01098814252666616, 0.004600577862184581, 0.011377768910919532, 0.005445176257926732, 0.008677331577145014, 0.009225240161982175, 0.005042716964760257, 0.009945430988596671, 0.006990029612656507, 0.01827400711501377, 0.006518817954134648, 0.008537075715339596, 0.014063975983381898, 0.0074988634030272805, 0.01944804080048925, 0.006456679081156653, 0.008761394660617044, 0.0038017491749662443, 0.03761135070342607, 0.0033952738124233294, 0.007499348032550552, 0.007038545119117952, 0.0072419628383620874, 0.015130853971079342, 0.003440499000494674, 0.0030783059077650254, 0.016025653643481883, 0.023300968204873422, 0.016564913176118808, 0.004616178536629922, 0.010211198716302123, 0.009579635220065393, 0.012592656113742487, 0.010098970958870086, 0.0036392584223820445, 0.009353431006599907, 0.008133092081132599, 0.019271113372638874, 0.013289776146287114, 0.024790137887250877, 0.021533203147463473, 0.0198423178022259, 0.016265772995178486, 0.0012075121549692993, 0.016264873890795208, 0.017120327909859772, 0.008965697496991196, 0.005070625006275076, 0.003259824578603002, 0.009654218200150716, 0.019446837089388283, 0.02282654283408265, 0.014922576969303372, 0.021105826009751763, 0.006555689317196447, 0.006846543088729413, 0.012385624261916384, 0.009343629265166964, 0.01788105156346237, 0.004202609602717985, 0.012524281060572684, 0.002423837058071331, 0.007632272022460165, 0.08403526991875405, 0.007318405860556596, 0.004654637105163784, 0.010672974742514815, 0.0024146137156501935, 0.011747632484563883, 0.010119423259268661, 0.016412611193631528, 0.009436769516147615, 0.0064562880661320375, 0.036550819898089316, 0.0019771903193784054, 0.006276847473233465, 0.0042036372745693365, 0.017751010713403594, 0.010889425604834106, 0.005868161582179096, 0.014889941026973474, 0.011161967864816325, 0.02041884446515687, 0.002887238658255228, 0.02192053975022864, 0.008534324279076827, 0.013801363037380839, 0.015578194689463998, 0.007704008659991086, 0.006374555025734845, 0.009527118786084338, 0.004677984993988139, 0.013686760475389019, 0.05342836332254428, 0.00959360407351778, 0.011119379543587276, 0.00794998789828078, 0.004015348075378643, 0.004463704052927642, 0.007114328121683411, 0.008495314261020667, 0.0031290600743916004, 0.008976399391647648, 0.013212694054467284, 0.015853073873254835, 0.004387003588787047, 0.010493587700571516, 0.015693420116185554, 0.01891398377813839, 0.030317615018305148, 0.01178025862483564, 0.009791753879134557, 0.0048866975179014845, 0.007336428658197474, 0.013204047778123278, 0.022062400079839607, 0.008136410626309225, 0.003376081286973204, 0.008059657756111118, 0.0047444069895256685, 0.010658050889334848, 0.0051665187407986205, 0.011192171886940194, 0.01121342160285276, 0.004822626063889039, 0.008942293757079915, 0.0046517555832173456, 0.0064019439255251194, 0.022986620871919418, 0.008510412274952613, 0.008418944527532949, 0.0157052840847683, 0.010820131553455212, 0.008459016686224641, 0.016847952436062224, 0.01478190362533569, 0.011114042579340275, 0.006515322566133181, 0.011771306592506336, 0.0028569384902151795, 0.016247816705189096, 0.008525444886229708, 0.006106762825808255, 0.008495487806445455, 0.008107241895833863, 0.02782632235249321, 0.02420878682834509, 0.008819177864608719, 0.005209856056111312, 0.01072808325715123, 0.013824486005163134, 0.003631150778734081, 0.017361496897741743, 0.00495113674600725, 0.03741164817760782, 0.006232081790731718, 0.03839060418025801, 0.0015971273978158933, 0.007286226926180552, 0.00890894936870609, 0.0029042195872696543, 0.01549434686300501, 0.01725151447448079, 0.004314562692510663, 0.010725466626472915, 0.020771962615435218, 0.05402022681185626, 0.008685576861305161, 0.0075164903075032275, 0.010420448055987798, 0.01739917813745633, 0.008825661315979687, 0.00823054950586038, 0.0115331966147555, 0.016033231853351388, 0.010894738554714602, 0.006969153263935732, 0.008163259801159074, 0.003994635083147386, 0.04229754067862887, 0.0035389652498721263, 0.0031829482628731053, 0.005510017333361793, 0.007578734275414717, 0.004584065979979277, 0.016859295864545494, 0.013390655095655258, 0.008116220686422822, 0.009973991301866408, 0.0039821077491327735, 0.023505909835879312, 0.006983075003395789, 0.0058244145003536005, 0.016923818328306518, 0.009829025329418801, 0.03709674534565398, 0.009369243108314641, 0.015524464354448354, 0.021050220287204865, 0.010451517519947266, 0.011451455802013153, 0.00860988747370975, 0.018423833689912225, 0.008021177285444357, 0.01675325480689046, 0.01611407067735927, 0.006614829253179778, 0.008849875774989777, 0.009924905411068174, 0.006661997877981776, 0.011714193304414583, 0.00269310245593515, 0.00632129345001722, 0.015204802567418664, 0.006491784230013218, 0.01105915630967429, 0.00438390145795347, 0.013215994669771408, 0.004516714549282096, 0.005518127828803971, 0.02050158296924401, 0.014672420009434427, 0.009314614396003888, 0.011864811019185777, 0.00497588384219489, 0.011163229661844527, 0.009415599596912396, 0.004916414776768511, 0.011203683079437358, 0.006895435796667516, 0.01471465752773961, 0.004015272386867674, 0.00636787604929773, 0.006010010007793971, 0.014323257760804517, 0.009816476613998749, 0.006507456901937802, 0.006308451339588009, 0.007554931442078396, 0.021695004051032085, 0.007837788066798385, 0.025868636885047088, 0.00805783973860107, 0.014730961419252524, 0.02832260190912457, 0.02302391439310086, 0.004486953584690855, 0.0013977065376227981, 0.007502759562358551, 0.005248604200036152, 0.003987702843360057, 0.013488426606849133, 0.006286396983430313, 0.018820764585410234, 0.007830320276208658, 0.00576615887028165, 0.010547195493716117, 0.007188092206232499, 0.012357032279620615, 0.011110565321089234, 0.005009508719965061, 0.013696918649262533, 0.0045937776767314975, 0.007952899416528453, 0.010384801897969491, 0.008423462865671718, 0.05910432837984118, 0.05759503486701906, 0.0049936416147043925, 0.0074818439830218264, 0.0038459080756130335, 0.014933405458999984, 0.032226774457342434, 0.0054019280972453084, 0.011963784766398023, 0.008658722658012994, 0.0055487125880641554, 0.008724131130260139, 0.01926567042400129, 0.009141180981095749, 0.015293712919565952, 0.0035472331285132898, 0.009181351291376814, 0.004639876724077566, 0.009552280195541791, 0.0077502091916565406, 0.008602183282960687, 0.0035855859203747964, 0.0126967330099651, 0.0094787369423565, 0.010442565066962197, 0.015683287463202578, 0.009174733530281082, 0.004840171245024339, 0.022117245394373593, 0.014143055751854625, 0.008766780631981783, 0.018901674148131587, 0.005242310763047863, 0.01862654724161948, 0.004887630752570839, 0.027778650715511867, 0.005310357367454435, 0.0021661386645721904, 0.02852371059631623, 0.00931900335083883, 0.017106688101586588, 0.004472820802043134, 0.03210532034698117, 0.0045687285450887434, 0.00288498678128708, 0.016779658233262124, 0.010405084950244054, 0.010519526549853527, 0.019350090563858658, 0.006261007772737502, 0.015434733209761544, 0.004436261214554542, 0.004580872874122906, 0.016482376733668206, 0.005604477996859054, 0.006967960860576567, 0.0064110468994382245, 0.007486814288021107, 0.0046738641938714855, 0.02653540599132084, 0.006985837093726149, 0.024296434636548292, 0.0037987163480625898, 0.014500807440721837, 0.005396692643335701, 0.01564793650378224, 0.015883084663544652, 0.007687133198832453, 0.007167044435137541, 0.0060692949358141955, 0.013195355794827265, 0.005508171509441673, 0.01265619678533658, 0.006847881005241743, 0.0070351455294123295, 0.0054334876010967495, 0.005940072341679478, 0.037042252123683284, 0.006111478871700819, 0.014597462643039825, 0.00449523753796384, 0.02719782488257688, 0.013246930990599868, 0.003592470943435183, 0.013791142861011033, 0.004806922129930899, 0.02854741480943776, 0.10320133036308142, 0.0069112240955651934, 0.007060361856169584, 0.017042855253431254, 0.0020458537453681854, 0.014802089768667063, 0.005163040772294471, 0.0019588670336182982, 0.006164608093014431, 0.006882887405025257, 0.0036385481383416386, 0.01466070411571086, 0.0037561753580795994, 0.0027727578770085524, 0.008277708604350493, 0.01090881901001485, 0.020106799343995046, 0.006722520315130713, 0.01857327775986769, 0.014957798810486384, 0.009639334888834127, 0.005996435027075495, 0.004651751958021488, 0.011981561372791476, 0.01151217119171497, 0.009621965013554654, 0.01773987257918481, 0.008409577345135222, 0.03771882053722111, 0.01450225348823078, 0.010618931732303645, 0.0034661452285745325, 0.00914534281212309, 0.010899815873719127, 0.0038085845507492195, 0.019838561982726974, 0.009396155671966699, 0.001713161140810189, 0.016610122740232144, 0.00359897416553722, 0.0037627551081663114, 0.01816143042643089, 0.008179642456733744, 0.033373092544344035, 0.04166043999372269, 0.013711819907006246, 0.014785530234450128, 0.004928172524060938, 0.012386216349234383, 0.008983680540824572, 0.008494815245354038, 0.015652141964068503, 0.0518572118705205, 0.0037311903234949566, 0.007830405482704007, 0.005488709290819771, 0.004769491629540061, 0.007762474916672909, 0.01217613252643714, 0.008888059557327688, 0.023736359881673705, 0.007113305837010011, 0.005613827157080686, 0.008373134489190125, 0.004306255558423527, 0.009505113183543835, 0.006578962360957122, 0.006334259897553785, 0.01077829706606126, 0.02863918442628383, 0.03428494307298952, 0.010755353148460198, 0.01175570131265149, 0.022148103400425997, 0.0050919892922411236, 0.009824750399365804, 0.007932845878993977, 0.009435705261860857, 0.011718709806068546, 0.007131150411527984, 0.023892131775104644, 0.023262013001505332, 0.051134846230692615, 0.015488243495593088, 0.008865127106933095, 0.006468065208609295, 0.010104501901374897, 0.008176616199809724, 0.0053484268712167245, 0.004697307261955575, 0.012576747082512156, 0.020811552193234453, 0.006619362909319898, 0.005176212113965631, 0.004022313684236179, 0.004029615894297649, 0.006859788958605986, 0.012070141270979821, 0.008922397736604033, 0.006095471137372807, 0.006642344718819276, 0.005913560051964305, 0.056609365113096465, 0.01770411586258033, 0.008914096570152834, 0.01477665406422784, 0.03445304462524846, 0.014150267668414018, 0.0059242078235838555, 0.005180388985327954, 0.014016628542633454, 0.008084046274091816, 0.0038028286099925025, 0.025413328229197216, 0.00428162276190651, 0.0033746127840182456, 0.010278599073917576, 0.020275542373006647, 0.023972785772504723, 0.007829168352291891, 0.007608653291303835, 0.00871585372279187, 0.02528613740788642, 0.01609967431171026, 0.010919195006880092, 0.00950435355557811, 0.002462812808694707, 0.0031286555699814927, 0.005748572041076113, 0.005660358625459776, 0.014718790444407387, 0.0195331802240129, 0.02020289572030488, 0.002878319613117856, 0.0035274368688242245, 0.025954628303534902, 0.0033111812844059383, 0.008011416357484995, 0.013071345421920342, 0.005148045828176145, 0.006810635359451589, 0.014750941609095753, 0.017245835930149316, 0.009182208374697621, 0.0022016354483985197, 0.16754732079771448, 0.015390625492636447, 0.015090553640143388, 0.010651973638068182, 0.0075671391537303866, 0.0024347766359858725, 0.014176074539121328, 0.016245349428696862, 0.017215551029279595, 0.014554863192699122, 0.005294451252179533, 0.007981104511267767, 0.005503336844294973, 0.006232688159937243, 0.0049967425462833495, 0.010436682628462588, 0.0039542141470687055, 0.014062826783752463, 0.03055683031260414, 0.012657900224072729, 0.006587205174776682, 0.06409145591075664, 0.009246913685215394, 0.0053634658574492275, 0.004314758665312729, 0.016622135721846645, 0.005592026126527344, 0.003951706369149543, 0.02756382388239726, 0.007011085819659457, 0.018229181869649617, 0.004333039434964303, 0.017019802723238536, 0.005094370000178605, 0.03516222204608889, 0.008956718829293822, 0.004497794048273499, 0.0033232561804838047, 0.006246241514805116, 0.00899138552511722, 0.008085096892007277, 0.026178633775873462, 0.011604506189093082, 0.01151888430002982, 0.003298371423415301, 0.02208149843235697, 0.00485639391419445, 0.017270203808109726, 0.013475115473957952, 0.018006920680365433, 0.004912246769731974, 0.010910116245574546, 0.028492647649701653, 0.02646192258586379, 0.0060762625964223835, 0.002706501697282511, 0.007729177539676675, 0.003237064107188781, 0.02627437785731235, 0.004785051566508991, 0.01246511548421913, 0.0018112136037419075, 0.003107074251004269, 0.009419986693524878, 0.021887113322313963, 0.009929573842477567, 0.01137414416362335, 0.01786310393031381, 0.03375420641709434, 0.01116382957711976, 0.008501765675754905, 0.009443623158454284, 0.008026612824015638, 0.022310075283454994, 0.0034604425673753214, 0.005489130418120476, 0.001962197849767472, 0.013634673433792703, 0.0286026984660326, 0.014730320402568397, 0.026574236261732404, 0.010768120527058635, 0.0058398869031228975, 0.006579848907079213, 0.010986864592744677, 0.03537099482527101, 0.0035496599415971103, 0.00487745701317399, 0.006772475154736185, 0.017008291253996687, 0.00932432157715466, 0.014114018287932413, 0.007130177591454413, 0.012027816381491538, 0.006317200123593102, 0.015865707418898957, 0.008659161483942262, 0.006767082021965143, 0.005125756160323727, 0.006625211309879159, 0.015832410001695826, 0.007219273784113375, 0.004403370951648685, 0.007237377016946993, 0.007103924527190248]
# 数値的に上下5%の値をみてみる
print(np.quantile(ls, [0.025, 0.975]))

# 1000回実施した結果のプロット
s.displot(data=ls)
plt.show()

df = pd.DataFrame(ls, columns=["shap_value"])
df["color"] = 1
print(df.head())

shap = pd.DataFrame([[0.013214498100209127, 2]], columns=["shap_value", "color"])
print(shap)
df = pd.concat([df, shap])

# Yをランダム化して1000回実施したSHAPと比較するヒストグラム
s.displot(data=df, x='shap_value', hue='color', multiple='stack')
plt.show()
'''

"""
df["x"] = 0
s.catplot(x='x', y='shap_value', data=df, kind='swarm', hue='color')
plt.show()

s.catplot(x='x', y='shap_value', data=df, kind='box', hue='color')
plt.show()
"""

MAD: Multi-Scale Anomaly Detection in Link Streams
EstebanBautista LaurentBrisson
IMTAtlantique,LUSSIDepartment, IMTAtlantique,LUSSIDepartment,
Lab-STICCUMRCNRS6285, Lab-STICCUMRCNRS6285,
Brest,France Brest,France
estbautista@gmail.com laurent.brisson@imt-atlantique.fr
CécileBothorel GrégorySmits
IMTAtlantique,LUSSIDepartment, IMTAtlantique,ComputerscienceDepartment,
Lab-STICCUMRCNRS6285, Lab-STICCUMRCNRS6285,
Brest,France Brest,France
cecile.bothorel@imt-atlantique.fr gregory.smits@imt-atlantique.fr
ABSTRACT ACMReferenceFormat:
Given an arbitrary group of computers, how to identify abnor- EstebanBautista,LaurentBrisson,CécileBothorel,andGrégorySmits.2024.
malchangesintheircommunicationpattern?Howtoassessifthe
MAD:Multi-ScaleAnomalyDetectioninLinkStreams.InProceedingsofthe
absenceofsomecommunicationsisnormalorduetoafailure? 17thACMInternationalConferenceonWebSearchandDataMining(WSDM
’24),March4–8,2024,Merida,Mexico.ACM,NewYork,NY,USA,9pages.
How to distinguish local from global events when communica-
https://doi.org/10.1145/3616855.3635834
tiondataareextremelysparseandvolatile?Existingapproaches
foranomalydetectionininteractionstreams,focusingonedge,
1 INTRODUCTION
nodesorgraphs,lackflexibilitytomonitorarbitrarycommunica-
tiontopologies.Moreover,theyrelyonstructuralfeaturesthatare A link stream is a set of triplets (𝑡,𝑢,𝑣) modeling that𝑢 and𝑣
notadaptedtohighlysparsesettings.Inthiswork,weintroduce interactedattime𝑡.Tripletsinalinkstreammayrepresentthat
MAD,anovelMulti-scaleAnomalyDetectionalgorithmthat(i) computer𝑢 sent a packet to computer𝑣 at time𝑡 or that bank
allowstoqueryforthenormality/abnormalitystateofanarbitrary account𝑢 made a transaction to account𝑣 at time𝑡. Detection
groupofobserved/non-observedcommunicationsatagiventime; oflikely/unlikelyinteractionsthatsuddenlydisappear/appearis
and(ii)handlesthehighlysparseanduncertainnatureofinteraction animportantsteptowardsidentifyingvariouseventsofcrucial
datathroughascoringmethodthatisbasedonanovelprobabilistic interest,suchasfinancialfrauds,networkattacks,orinfrastruc-
andmulti-scaleanalysisofsub-graphs.Inparticular,MADis(a) turefailures.Forexample,considerthecaseillustratedinFigure1,
flexible:itcanassessifanytime-stampedsubgraphisanomalous, depictinginteractionsbetweenserversandusersrequestingtheir
makingedge,nodeandgraphanomaliesparticularinstances;(b)in- services.Itisrathernormalthattheserversfrequentlyexchange
terpretable:itsmulti-scaleanalysisallowstocharacterizethescope trafficbetweenthemandalsowithsomeregularusers.Figure1
andnatureoftheanomalies;(c)efficient:givenhistoricaldataof representssuchusersandserversthatfrequentlyinteractinblue,
length𝑁 and𝑀observed/non-observedcommunicationstoana- whileitrepresentsusersthatconnectlessfrequentlyingrey.Ifat
lyze,MADproducesananomalyscoreinO(𝑁𝑀);and(d)effective: agiventimethecommunicationpatterndepictedintheleftpanel
itsignificantlyoutperformsstate-of-the-artalternativestailoredfor isobserved,thesituationcanbelabeledasnormalgiventhatthe
edge,nodeorgraphanomalies. observedinteractionsonlyconcernentitiesthatusuallyinteract
together.Yet,iftheobservationcorrespondstotheonedepictedin
CCSCONCEPTS therightpanel,suchchangeinthecommunicationpatternmaybe
anindicationofafailureoranattack.Anotherexamplecanbea
•Informationsystems→Datastreammining;Socialnetworks;
bankaccountthatsuddenlystartstomaketransactionstoseveral
Trafficanalysis;•Securityandprivacy→Intrusion/anomaly
unexpectedaccounts.Suchbehaviormaybeindicativeofafraud.
detectionandmalwaremitigation.
Tospottheaforementionedevents,numerouslinkstream-based
anomalydetectionalgorithmshavebeenproposedinrecentyears.
KEYWORDS
Inanutshell,suchalgorithmscanbeseenasblackboxesthatreceive
anomalydetection,temporalnetworks,modelinterpretability twoinputs,aqueryandacontext,andanswertothequestion:how
abnormalisthequerygiventhecontext?Theessentialdifferences
Permissiontomakedigitalorhardcopiesofallorpartofthisworkforpersonalor betweentheexistingalgorithmsare:(i)thetypesofqueriestheyac-
classroomuseisgrantedwithoutfeeprovidedthatcopiesarenotmadeordistributed
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation cept;(ii)thedefinitiongiventothenotionofanomaly;and(iii)how
onthefirstpage.Copyrightsforcomponentsofthisworkownedbyothersthanthe thecontextisexploited.Forexample,numerousalgorithmsaccept
author(s)mustbehonored.Abstractingwithcreditispermitted.Tocopyotherwise,or
time-stampededgesasqueries,yettheydifferinthecriterionused
republish,topostonserversortoredistributetolists,requirespriorspecificpermission
and/orafee.Requestpermissionsfrompermissions@acm.org. tolabelaqueryasabnormal:somedoitwhenthequeryimpliesa
WSDM’24,March4–8,2024,Merida,Mexico suddenchangeinedgecounts[1],nodeembeddings[2],orwalk
©2024Copyrightheldbytheowner/author(s).PublicationrightslicensedtoACM. statistics[3],whileothersdoitwhenthequerycannotbewellpre-
ACMISBN979-8-4007-0371-3/24/03
https://doi.org/10.1145/3616855.3635834 dictedfromthepast[4,5].Algorithmsaddressingcoarserresolution
38

WSDM’24,March4–8,2024,Merida,Mexico EstebanBautista,LaurentBrisson,CécileBothorel,&GrégorySmits
acceptsarbitrarytime-stampedsubgraphsasqueries;and(ii)labels
queriesasabnormaliftheyhave/lackmanyunlikely/likelyinter-
actions,thusallowingtohandlethedatauncertaintyandsparsity.
MADisbasedonanovelmulti-scaleprobabilisticanalysisforsub-
graphsthatessentiallypermitstomapaquerysub-graphintoaset
ofrandomvariablesfromwhichitispossibletoidentifythescale(s)
atwhichaqueriedsub-graphcannotbewellexplainedfromitspast
activity.Propertiesoftheproposedcontributionareasfollows:
• Flexibility:MADcanbeusedtodetermineifanyarbitrarytime-
stampedsubgraphisanomalous,thusmakingedge,nodeand
graphanomalydetectionparticularinstancesofourapproach.
• Interpretability:thedevelopedmulti-scaleanalysisallowstoiden-
tifythescale(s)andthenatureoftheevent(s)makingaquery
Figure 1: Examples of normal and abnormal communica-
abnormal.
tionpatterns.Agroupofserversusuallyexchangebetween
• Efficiency:givenaqueryofsize𝑀 andahistoricalcontextof
themandwithsomeexternalusers.Historicaldatamaysug-
duration𝑁,MADanswersinO(𝑁𝑀).
gestthatinteractionsbetweenblue-colorednodesarehighly
• Effectiveness:MADsignificantlyoutperformsstate-of-the-artal-
likely.Thesuddenhaltoflikelytrafficandtheemergence
ternativesforedge,nodeandgraphanomalydetectioninthetasks
ofunlikelyonemaybeanindicationofanattack(hackers
ofidentifyingcommunicationsthatabnormallyappear,disappear
havetakencontrolofthemachines)orafailure(engineers
orgetredirected.
aretroubleshooting).
queries,liketime-stampednodes[6,7]orentiregraphsnapshots 2 NOTATIONSANDRELATEDWORKS
[8–11],havealsobeenproposed.Thesealgorithmsalsovaryinthe
2.1 Notations
waytheydefineananomalyandusethecontext.Namely,nodes
maybedeemedabnormaliftheysuddenlychangetheircentrality
Let𝑉 beasetofvertices,𝑇 refertonaturalnumbers,E =𝑉 ×𝑉
[6]orcommunicationcounts[7],whilegraphsmaybeconsidered
denotearelationspace,and𝜙 ⊆Ebeanarbitrarysetofrelations
abnormaliftheyhavesuddendensifications[9],spectrumchanges
ofsize|𝜙| =𝑀.Adiscrete-timelinkstreamisdenotedbytheset
[10],orcommunityre-configurations[11],tolistsomeexamples.
𝐿 ⊆𝑇 ×E.Therestrictionof𝐿toatimeinterval [𝑡1,𝑡2] andset
Inspiteofnumeroussuccesses,existinganomalydetectionalgo- of relations𝜙 is expressed as𝐿(𝑡1 : 𝑡2,𝜙) = {(𝑡,𝑢,𝑣) ∈ 𝐿: 𝑡1 ≤
rithmsremainnotfullysatisfactory.Inparticular,thefactthatthey 𝑡 ≤ 𝑡2,(𝑢,𝑣) ∈ 𝜙}. The case𝑡1 = 𝑡2 corresponds to a slice, or
snapshot,of𝐿.Strictlyspeaking,theinteractionsofasliceremain
onlyaccepttime-stampedsubgraphsofaspecificformasqueries
time-stamped,yetinsomesituationsitisusefultostripthetime
(eitheredges,nodes,orentiregraphs)makesthemtoorigidfor
referencefromthemsothattheycanbeconsideredastheedgesof
severalreal-worldusecases.Inmanysituationsitisdesirableto
agraph,thatisindependentoftime.Therefore,giventherestriction
questionanalgorithmifanarbitrarycommunicationtopologybe-
havesabnormally:takeforinstancetheexampleofFigure1,where
sets𝑡1=𝑡2=𝑡and𝜙,welet𝐿(𝑡:𝑡,𝜙)refertothesliceof𝐿inwhich
onewantstotrackthecommunicationsbetweenaspecificgroup
interactionsremaintime-stamped,whilewelet𝐿(𝑡,𝜙)denotethe
thecaseinwhichthetime-stampsarestriped-out.Relationsare
ofserversandusers;ortakethecasewhentransactionsbetween
aspecificgroupofbankaccountssuspectedtobelongtoacrimi-
considereddirected,hence(𝑢,𝑣)≠(𝑣,𝑢).Moreover,forthesakeof
notationlightness,therelationsemergingfromnode𝑢aredenoted
nalorganizationhavetobemonitored.Tohandlethesedifferent
cases,anultimatesolutionistodisposeofanalgorithmcapable
byE𝑢 ={(𝑢,𝑣):𝑣 ∈𝑉}.
Inthiswork,weextensivelyuseindicatorfunctionstocharac-
torespondtoqueriesconsistingofarbitrarytime-stampedsub-
terizesubsetsofaset:abinaryfunctionindicatingwhichelements
graphs.Somealgorithmsforanomaloussubgraphdetectionhave
from the set belong or not to the subset. The concept plays an
beenproposed[12–14],yetsuchapproachesautomaticallysearch
importantroleinthisworkgiventhatalgorithmsworkingdirectly
for subgraphs that meet some criteria, like being a dense com-
munity,thuspreventingusersfromqueryingthealgorithmswith
withaset𝐴⊆𝐵cannotmakedecisionsbasedontheelementsof𝐵
notincludedin𝐴,astheyignorethem;whilealgorithmsworking
arbitrarysubgraphs.Itisalsoworthnoticingthatmostalgorithms
withtheindicatorfunctionknowsuchinformation.Formally,the
targetingcoarse-grainqueriesrelyonanomalydefinitionsthatdo
notsatisfactorilyaccountforthehighlyuncertainandsparsena-
indicatorfunctionof𝐴⊆𝐵isdenotedby1
𝐴
𝐵 :𝐵→{0,1},where
tureoftemporalinteractions.Forexample,manyworksrelyon 1𝐵(𝑥)=1if𝑥 ∈𝐴andzerootherwise.Thus,theindicatorfunction
𝐴
anomalydefinitionsthatarebasedonchangesofnodecentrality, ofalinkstreamisgivenby1𝑇×E(𝑡,𝑢,𝑣) = 1if (𝑡,𝑢,𝑣) ∈ 𝐿 and
𝐿
walkstatistics,spectralpropertiesorcommunitystructures.These zerootherwise.Also,withtheaimofnotationlightness,wedenote
definitionsimplicitlyassumedenselinkstreamsthatslowlyevolve, thesumofafunction 𝑓 : 𝐵 → Roverasub-domain𝐶 ⊆ 𝐵 by
whichmaybeunrealisticinnumeroussituations. 𝑓(𝐶)=(cid:205)
𝑐∈𝐶
𝑓(𝑐).Hence,1
𝐴
𝐵(𝐶)=(cid:205)
𝑐∈𝐶
1
𝐴
𝐵(𝑐)for𝐶 ⊆𝐵.Lastly,
Theaimofthisworkistoaddressthelimitationslistedabove. QreferstoaqueryandH toahistoricalcontext.ThenatureofQ
WeintroduceMAD,anovelanomalydetectionalgorithmthat(i) andH dependsontheconsideredalgorithm,asdetailednext.
39

MAD:Multi-ScaleAnomalyDetectioninLinkStreams WSDM’24,March4–8,2024,Merida,Mexico
Algorithm Query(Q) Context (H)
MIDAS[1] 𝐿(𝑡,(𝑢,𝑣)) 𝐿(0:𝑡−1,(𝑢,𝑣))
F-FADE[2] 𝐿(𝑡,(𝑢,𝑣)) 𝐿(0:𝑡−1,E)
SEDANSPOT[3] 𝐿(𝑡,(𝑢,𝑣)) 𝐿(0:𝑡−1,E)
DEGOD[17] 𝐿(𝑡,E𝑢) 𝐿(0:𝑡,E)
Figure2:Unifiedviewofanomalydetection.Algorithmsaim
DYNANOM[6] 1E𝑢 𝐿(𝑡−1:𝑡,E)
toassesstheabnormalityofaqueryinagivencontext. 𝐿(𝑡,E𝑢)
DSEDN[15] 1E𝑢 𝐿(0:𝑡,E)
𝐿(𝑡,E𝑢)
2.2 RelatedWorks ANOMRANK[8] 1E 𝐿(𝑡−2:𝑡,E)
𝐿(𝑡,E)
Theultimatealgorithmforlinkstream-basedanomalydetectionis
SPOTLIGHT[9] 𝐿(𝑡,E) 𝐿(0:𝑡−1,E)
onethatreceivestwoarbitrarysub-linkstreamsasinputs,constitut-
ingaqueryQandacontextH,andthatrespondstothequestion: LAD[10] 1
𝐿
E
(𝑡,E)
𝐿(𝑡−𝑘 :𝑡,E)
canthequerybeexplainedfromtheprovidedcontext?SeeFig-
Table1:Summaryofthemostrepresentativeworksproposed
ure2foranillustration.Noalgorithmisabletoexploreallthe
intheliterature.
possiblewaysinwhichanarbitraryquerymaybeexplainedfrom
anarbitrarycontext.Thus,existingalgorithmsessentiallynarrow
thesearchby(i)focusingonqueriesandcontextsthatadhereto
aspecificform;and(ii)establishingaspecificnormalitycriterion
thatthequerymustpossessinordertobeconsideredasexplained model.DEGOD[17]usespastinteractionstocomputethedegree
bythecontext.Asaresult,thereisarichvarietyofapproachesthat distributionofnodesovertime.Itlabelsaqueryasabnormalifit
focusondifferentcombinationsofinputsandnormalitydefinitions. causesthecurrentdegreedistributiontonotmatchthepast.
Webrieflyreviewproposedalgorithms,structuredaccordingto Graphanomaly. Thesealgorithmsacceptentiretime-stamped
typesofqueriestheyhandle.Table1presentsasummary. slicesasqueries.Inparticular,ANOMRANK[8]usesalocalcontext
Edgeanomaly. Algorithmsinthiscategoryrespondtoqueries totracktheevolutionofPageRankscoresofvertices.Itlabelsa
consistingoftime-stampededges.Thedifferencebetweenthemlies queryasabnormalifitsfirstderivativesindicateadrasticchange.
inthecriteriaemployedtoconsideraqueryasabnormal.Namely, SPOTLIGHT[9]measuresthedensityofrandompartitionsofhis-
MIDAS[1]usesthehistoryofthequeryedgetopredictitsfuture toricaldatatocomputeasetofreferencevectors.Itlabelsaquery
activity.Itlabelsthequeryasabnormalifitappearsinaperiodpre- asabnormalifitscorrespondingvectorisanoutlierwithrespectto
dictedtobeoflowactivity.F-FADE[2]useshistoricalinteractions thecomputedreferences.LAD[10]useslong-termandshort-term
tocomputeanodeembeddingthatexplainsedgefrequencies.It contextstopredictthespectralshapeofthequeryslice.Itlabels
considersaquerytobeabnormaliftheobserveddistancebetween thequeryasabnormalifitsspectrumisfarfromtheexpectedval-
edgeendsisimplausible.SEDANSPOT[3]usesthepasttoestimate ues.CADENCE[11]usespastinteractionstoidentifycommunity
agraphofstablecommunities.Thequeryisabnormalifitsinclu- structuresthatarestableovertime.Itlabelsaqueryasabnormal
siontosuchgraphbreaksrandomwalksstatistics.AER-AD[5] ifitimpliesacommunityreconfiguration.ODGS[18]useshistori-
usesthepasttotrainarecurrentneuralnetworkforlinkprediction. caldatatofitacommunity-orientedgenerativemodel.Aqueryis
Itlabelsaqueryasabnormalifitisnotpredictedwell.PIKACHU abnormalifitcontainsmanyinter-communityedges.
[4]extractstemporalrandomwalksfromhistoricalinteractionsto Problemstatement. Asitcanbeseen,proposedalgorithms
trainanencoderforlinkprediction.Itlabelsaqueryasabnormal essentiallyfixatime𝑡 andagroupofrelations𝜙 ofaspecificform:
ifitisnotpredictedwell. 𝜙 =(𝑢,𝑣),𝜙 =E𝑢 ,or𝜙 =Eandaddress𝐿(𝑡,𝜙)asqueries.While
Nodeanomaly. Thesealgorithmsaddresstime-stampednodes thisallowstotacklemanyabnormalevents,thefactthat𝜙mustpos-
asqueries,whichcanberepresentedbytheirsetofincidentedges sessaspecificform(eitheredges,nodes,orgraphs)drasticallylimits
orbytheindicatorfunctionofsuchset.DYNANOM[6]usesashort theflexibilityandeffectivenessofalgorithmsinmanyapplication
termcontexttomonitortheevolutionofPageRankscores.Itla- scenarios,liketheoneillustratedinFigure1whereonemaybein-
belsaqueryasabnormalifitsPageRankscoredrasticallychanged. terestedinmonitoringanarbitrarygroupofcommunications.This
BADSN[7]usesthehistoricalinteractionsofthequerynodeto callsforanalgorithmthatallowstoset𝜙 asanarbitrarysubgraph.
modeltheprobabilityofobservingitwithagivendegree.Itlabels Somealgorithmshavebeenproposedtoaddressthecaseinwhich𝜙
thequeryasabnormaliftheobserveddegreeisunlikelyaccording isacommunityclique[12–14].Yet,suchalgorithmsautomatically
tothemodel.DSEDN[15]usespastinteractionstotrainanauto- searchforagroupofcandidatecommunities,thuspreventingusers
encoderthatembedsnodesinawaythatstablestructuresover toquerythealgorithms.Moreover,westressthatthecommunity
timeformclusters.Itlabelsaqueryasabnormalifitisanoutlier criterionassumesslicesofdenselinkstreams,whichisunrealistic
intheembeddingspace.GEABS[16]leverageshistoricalinterac- inmostreal-worldinteractiondata.Thus,theaimofthisworkisto
tionstofitacustom-madegenerativemodelthatjointlyaccounts addressthesetwosituations:toproposeanalgorithmthatallows
forcommunitystructureandnodepopularity.Itlabelsaqueryas toset𝜙 asanarbitrarysubgraphandthatdoesitbytakinginto
abnormalifitscommunitymembershipisunstableaccordingtothe accountthehighlyuncertainandsparsenatureofinteractions.
40

WSDM’24,March4–8,2024,Merida,Mexico EstebanBautista,LaurentBrisson,CécileBothorel,&GrégorySmits
3 PROPOSEDALGORITHM However,whilesimple,thisapproachisunsatisfactoryforanom-
alydetectionforthefollowingtworeasons:(i)small-scaleanom-
Inthissection,weintroduceMAD,anovelsolutiontoanomaly
alieshavelittleimpactin(1);and(ii)expectedobservationsarenot
detection.Inadditiontoaddressingarbitrarytime-stampedsub-
graphsasqueries,MADalsotakesintoaccountthat(i)real-world rankedasthemostprobable(hencenormal)by(1).Noticethatafew
interactionsarehighlydynamicanduncertain;and(ii)anomalous abnormaledgesmaynotdrivethevalueof(1)sufficientlylowto
eventscanbeofdifferentscales.Indeed,itisnormalthatpersonsor beconsideredasaclearanomaly.Moreover,consideracasewhere
|                                |     |     |     |                        |     | 𝜙 ={𝑒1,𝑒2,𝑒3}and𝑃(𝑒 | 𝑖)=1/3forall𝑒 | .Accordingto(1),themost |     |
| ------------------------------ | --- | --- | --- | ---------------------- | --- | ------------------- | ------------- | ----------------------- | --- |
| computerscommunicatingatatime𝑡 |     |     |     | donotcommunicateat𝑡+1, |     |                     |               | 𝑖                       |     |
probableobservationforthissettingistheemptysubgraph,i.e.
eventhoughitmaynotbesurprisingifitoccurs,thusmakingthe
when𝜙ˆ
datahighlydynamicanduncertain.Moreover,ahackerintrusion =∅.Thisisundesirableastheemptysubgraphisnotthe
mayreflectatthescaleofafewcommunications,whileaninfras- oneexpectedtoappearfromsuchprocess:onesuccessisexpected.
tructurefailuremayinvolvemostofthem.MADisasolutionto Thisraisesthequestionofhowtofindmeaningfulanomalyscores
thatallowtospoteithersmallorlargescaleanomalies.
thesesituations.Itisdevelopedasfollows.Firstly,Section3.1devel-
Interestingly,analternativeistouserandomvariablesmeasuring
opsascoringfunctionbasedonanovelprobabilisticandmulti-scale
analysisofsubgraphs.Itassumesthatinteractionsaregeneratedby propertiesoftheanalyzedsubgraphs,liketheirnumberofactive
relations|𝜙ˆ
aknownrandomprocessanddefinesasetofmulti-scalerandom |.Theadvantageofusingrandomvariablesisthatwe
variablesthatallowtospotqueriesdeviatingfromthepastinaway cancharacterizethevaluestheytakewhentheyarecomputedon
subgraphsgeneratedbytheunderlyingprocess.Thus,whenrare
thatcannotbeexplainedfromtheusualuncertainty.Section3.2
valuesareobserved,theunderlyingsubgraphcanbeconsidered
addressestheproblemofestimatingtheaboverandomprocessfrom
thehistoricalcontext.Astationarytestisdevelopedtoautomati- anomalous.Thechallengewiththisapproachliesinhowtodefine
callyidentifythewindowlengthinwhichtheprocesscanbebest meaningfulrandomvariablesthatmeasurepropertiesthatcanspot
estimated.Insum,MADisamulti-scaleanomalydetectionalgo- alltargetedanomalies.Forinstance,|𝜙ˆ |isausefulrandomvariable
𝜙 thatallowstoreadilyspotdensificationorsparsificationeventsby
| rithmthatacceptsQ=1 |     |            | andH | =𝐿(𝑡−𝑁:𝑡−1,𝜙)asinputs, |     |                                               |     |     |     |
| ------------------- | --- | ---------- | ---- | ---------------------- | --- | --------------------------------------------- | --- | --- | --- |
|                     |     | 𝐿 ( 𝑡, 𝜙 ) |      |                        |     | usingitsexpectedvalueasareference.However,|𝜙ˆ |     |     |     |
|aloneisnot
| where𝜙isanarbitrarysu |     | b g r a | phand𝑁 | isthecontextduration,and |     |     |     |     |     |
| --------------------- | --- | ------- | ------ | ------------------------ | --- | --- | --- | --- | --- |
enoughtodetectallanomalies:aneventwhere𝑘likelyrelations
| returnsananomalyscoredenotedby𝑠𝑐𝑜𝑟𝑒(Q).H |     |     |     |     | isintroduced |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- |
areinactiveand𝑘unlikelyonesareactivewouldbenormalunder
hereasatimewindowintheinteractionsthatprecedethequery,
|     |     |     |     |     |     | thecriterionof|𝜙ˆ | |.Inthefollowing,weaddressthischallengeby |     |     |
| --- | --- | --- | --- | --- | --- | ----------------- | ----------------------------------------- | --- | --- |
butMADcanindifferentlyconsideranytimewindowtakeninthe
history,whichisespeciallyusefultocaptureperiodicallystationary introducingamulti-scaleanalysisofsubgraphs.Itdefinesagroup
of𝑀 =|𝜙|randomvariablesthatquantifyandcomparetheactivity
interactionpatterns.
ofthequerysubgraphatmultiplescales.Wethencharacterizetheir
distributionfornormalqueriesinordertospotthescaleandgroup
3.1 Amulti-scaleanalysisofsub-graphs ofrelationsthatmakeanobservationanomalous.
| Ourgoalistoassessifthebinarystate(activeorinactive)ofa |     |     |     |     |     |     |     |     | 𝜙   |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Letusbeginthedevelopmentofourmulti-scaleanalysisof1
𝜙ˆ
| groupofrelations𝜙 |     | ⊆Eattime𝑡 | isabnormal.Forsimplicity,we |     |     |     |     |     |     |
| ----------------- | --- | --------- | --------------------------- | --- | --- | --- | --- | --- | --- |
denotethesetofactiverelationsby𝜙ˆ bymakingtwoassumptionsaboutitsdomain.Firstly,weassume
|                  |     |        |                               | := 𝐿(𝑡,𝜙) | anditsbinary |                                  |     |                              |        |
| ---------------- | --- | ------ | ----------------------------- | --------- | ------------ | -------------------------------- | --- | ---------------------------- | ------ |
|                  |     |        |                               |           |              | 𝑀 =|𝜙|tobeapoweroftwo.If𝜙        |     | lacksrelationsforthistohold, |        |
| statefunctionby1 | 𝜙   | :=1 𝜙  | .Asmentionedabove,animportant |           |              |                                  |     |                              |        |
|                  |     |        |                               |           |              | thenweassumethatvirtualelements𝑒 |     | ofprobability𝑃(𝑒             | 𝑖) = 0 |
|                  | 𝜙ˆ  | 𝐿(𝑡,𝜙) |                               |           |              |                                  |     | 𝑖                            |        |
propertytotakeintoaccountconcernstheuncertaintyrelativeto areaddedinto𝜙 untiltheassumptionholds.Westressthatthe
inclusionofthesevirtualrelationsisofpuremathematicalconve-
| eachinteractioninvolvedin𝜙.Thus,𝜙 |     |     |     | isinterpretedastheresult |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- |
ofarandomprocess.Inparticular,weassumetheexistenceofa nienceandtheydonothinderouranalysisasthoseelementsare
function𝑃 : 𝜙 → [0,1] sothatthestateofeach𝑒 ∈ 𝜙 isseen alwaysswitched-offin1 𝜙 ,whichisinagreementwiththeirnull
|                                                       |     |     |     |     | 𝑖   |                         | 𝜙ˆ                           |     |     |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | ----------------------- | ---------------------------- | --- | --- |
| astheresultofrunningaBernoullitrialwithprobability𝑃(𝑒 |     |     |     |     | 𝑖): |                         |                              |     |     |
|                                                       |     |     |     |     |     | probability.Secondly,we | assumethatthattheelementsof𝜙 |     | are |
𝜙
ifthetrialissuccessfulthen1 (𝑒 𝑖) = 1andzerootherwise.In indexedindecreasingorderoftheirprobability.Itisassumedthat
𝜙ˆ
|     |     |     |     |     |     |     |     | 𝑀)forall𝑒 | ∈𝜙.Based |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | -------- |
thecomplexnetworksterminology,thisisequivalenttointerpret 𝑃(𝑒1) ≥···≥𝑃(𝑒 𝑖) ≥𝑃(𝑒 𝑖+1) ≥···≥𝑃(𝑒 𝑖
that1 𝜙 isgeneratedbyanextendedErdös-Rényimodelwherethe ontheseassumptions,thefirststepofourmulti-scaleanalysiscon-
𝜙ˆ
sistsinrecursivelypartitioning𝜙atdifferentresolutionscales.To
| p ro b a b il i | t ie s o fe d g e | s a r e i n d i v | i d u a ll y | t u n e d . T h e n , w  | e co n s id e r a n   |                         |     |                                |     |
| --------------- | ----------------- | ----------------- | ------------ | ------------------------ | --------------------- | ----------------------- | --- | ------------------------------ | --- |
|                 |                   |                   |              |                          |                       | doit,wesetaninitialsetE | (0) | =𝜙thatwesplitinhalvesaccording |     |
|                 | 𝜙                 |                   |              |                          |                       |                         | 0   |                                |     |
| ob se r v a t i | o n 1 a s a b     | n o r m a l i f   | i t i s u n  | li k e l y t o h a v e b | e en g e n er a t e d |                         |     |                                |     |
𝜙ˆ totheprobabilityofitselements:thetop-halflikelyrelationsare
bysuchprocess.Throughouttherestofthissubsection,weassume assignedtoasetE (1) ={𝑒1,...,𝑒 𝑀}andthebottom-halfonesto
0
| that𝑃isknownandthatitaccuratelymodelsnormality.Inpractice, |     |     |     |     |     |     |     | 2   |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(1)
wemustestimate𝑃 fromH.Section3.2addressessuchproblem. asetE ={𝑒 𝑀+1 ,...,𝑒 𝑀}.Thisrecursivepartitioningisapplied
|     |     |     |     |     |     | 1 2 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Givenourassumedprobabilisticmodel,asimpleandstraight- (log 2(𝑀))
|     |     |     |     |     |     | untilsingletonsareobtainedE |     | =𝑒 .Foravisualreference, |     |
| --- | --- | --- | --- | --- | --- | --------------------------- | --- | ------------------------ | --- |
|     |     |     |     |     |     |                             | 𝑖   | 𝑖                        |     |
forwardwaytospotunlikelyrealizationsatthesubgraphlevel seethebinarytreestructuredisplayedinFigure3,wheretheroot
| consistsincomputingtheexactprobabilityofobserving𝜙,which |     |     |     |     |     | (0)                                                     |     |     |     |
| -------------------------------------------------------- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- |
|                                                          |     |     |     |     |     | nodeisE andthechildrennodesrepresentthepartitionedsets. |     |     |     |
| isgivenby:                                               |     |     |     |     |     | 0                                                       |     |     |     |
Algebraically,thepartitioningruleis:
𝜙
𝑃𝑟(1 )=Π 𝑃(𝑒 𝑖)×Π 𝑒𝑗∈𝜙\𝜙ˆ(1−𝑃(𝑒 𝑗)). (1) (ℓ) ( ℓ+1) ( ℓ+ 1 ),
|     | 𝜙ˆ  | 𝑒𝑖∈𝜙ˆ |     |     |     |     | E =E | ∪E        | (2) |
| --- | --- | ----- | --- | --- | --- | --- | ---- | --------- | --- |
|     |     |       |     |     |     |     | 𝑘 2  | 𝑘 2 𝑘 + 1 |     |
41

MAD:Multi-ScaleAnomalyDetectioninLinkStreams WSDM’24,March4–8,2024,Merida,Mexico
where Theinterestingconnectionbetweenthismulti-scaleanalysisand
E 𝑘 (ℓ) = (cid:26) 𝑒 𝑖 ∈𝜙 : 𝑘 2 𝑀 ℓ +1≤𝑖 ≤ (𝑘+ 2ℓ 1)𝑀(cid:27) . (3) m th o e m as e s n u t m s e o d f r 𝑠 a a n n d d om 𝑤 p (ℓ r ) oc c e a s n s b is e t e h x a p t r t e h s e s fi ed rst in an te d r s m ec s o o n f d 𝑃 t . h T eo h r i e s t i i s ca a l
𝑘
Thus,theprocedureabovepartitions𝜙intodisjointsubgraphsat crucialpropertyforanomalydetectionasitallowstocharacterize
differentresolutions,asindicatedbythesuper-scriptℓ.Particularly, therangesofvaluesthat𝑠and𝑤(ℓ) normallytakewhentheyare
𝑘
2ℓ partitionsariseatlevelℓandtheysatisfythefollowingcrucial computedonrealizationsgeneratedby𝑃.Ournextresultstates
property:norelationcontainedinE (ℓ) ismoreprobablethanthe thisconnection.
𝑘+1
relationscontainedinE 𝑘 (ℓ).Inthesecondstepofouranalysis,we Proposition2. LetE[·] and𝜎2[·] denotetheexpectationand
leveragethispropertybydefiningrandomvariablesthatcompare 𝜙
theactivityofE (ℓ) withthatofE (ℓ).Thisisanaturalapproachto varianceoperators,respectively.Ifthefunctions1 𝜙ˆ aredrawnfrom
spotanomaliesa
𝑘
tmultiplescales,a
𝑘
s
+1
weknowthat,byconstruction, thegenerativemodeldefinedabove,itholdsthat:
E (ℓ) shouldbemoreactivethanE (ℓ).Inpreciseterms,wedefine (a) E[𝑠] = √ 1 𝑃(𝜙),
th 𝑘 efollowingsetofrandomvariab 𝑘 le + s 1 : (b) E[𝑤(ℓ) ] = 𝑀 √ √ 2ℓ (cid:104) 𝑃 (cid:16) E (ℓ+1)(cid:17) −𝑃 (cid:16) E (ℓ+1)(cid:17)(cid:105) ,
𝑘 𝑀 2𝑘 2𝑘+1
𝑠 = √ 1 1 𝜙 (𝜙), (4) (c) 𝜎2[𝑠] = 𝑀 1 (cid:205) 𝑒𝑖∈𝜙 𝑃(𝑒 𝑖)[1−𝑃(𝑒 𝑖)],
and
𝑀 𝜙ˆ (d) 𝜎2[𝑤 𝑘 (ℓ) ] = 𝑀 2ℓ (cid:205) 𝑒𝑖∈E
𝑘
(ℓ) 𝑃(𝑒 𝑖)[1−𝑃(𝑒 𝑖)].
√
𝑤 𝑘 (ℓ) = √ 𝑀 2ℓ (cid:20) 1 𝜙 𝜙ˆ (cid:16) E 2 ( 𝑘 ℓ+1)(cid:17) −1 𝜙 𝜙ˆ (cid:16) E 2 ( 𝑘 ℓ+ + 1 1 )(cid:17)(cid:21) . (5) tha F t r a om ran th do e m Ch p e ro b c y e s s h s e p v r i o n d e u q c u e a s l o it b y s , e w rv e a k ti n o o n w so t f h a at ra th n e do p m ro v b a a r b i i a l b it l y e
thatare𝜆standarddeviationsawayfromitsexpectationcannotbe
forall𝑘andℓ.Intotal,(4)and(5)define𝑀randomvariablesasthere
are2ℓ setsassociatedtoℓandthisonerunsfrom0tolog 2(𝑀)−1. l
i
a
n
r
c
g
r
e
e
r
as
th
e
a
q
n
u
1
a
/
d
𝜆
r
2
a
.
ti
H
ca
e
l
n
ly
ce
i
,
n
ou
th
r
e
su
n
s
u
p
m
ici
b
o
e
n
r
a
o
b
f
o
s
u
ta
t
n
an
da
o
r
b
d
se
d
r
e
v
v
a
i
t
a
i
t
o
i
n
on
s
s
ho
th
u
a
ld
t
Therefore,bydoingthisanalysiswedonotchangethesizeofthe
itsrandomvariablesvaluesareawayfromthemean.Basedonthis
problem:wetransitionfromanalyzingthestateof𝑀relationsin
property,wecandefineananomalyscoreforeachrandomvariable
𝜙
1 to𝑀randomvariables.Moreover,itisworthnoticingthatthe
𝜙ˆ givenastheinverseofitsChebyshevbound.If𝑥 𝑖 denotesthe𝑖-th
randomvariablescanbecomputedinO(𝑀)usingthebinarytree randomvariable,thenitsanomalyscoreisgivenasfollows:
𝜙
showninFigure3:bysetting1 intheleaves,successiveparents
𝜙ˆ score(𝑥 𝑖)=(𝑥 𝑖 −E[𝑥 𝑖]) 2 /𝜎2 [𝑥 𝑖]. (7)
comparetheactivityofrelationsappearingintheirleftandright
branches,producingthedesiredrandomvariables. Ifweaimtofavorinterpretability,wecanreturnthe𝑀anomaly
Concerningtheanalysisoftherandomvariables,noticethat,on scoresaboveastheoutputofthealgorithm,allowingauserto
theonehand𝑠correspondstoanormalizedversionof|𝜙ˆ |,which, identifywhichpartsofthequerysubgraphareattheoriginofan
asmentionedpreviously,isrelevanttodetectdensificationorspar- anomaly.Forsimplicity,wereturnasingleanomalyscoresumma-
sificationevents.Andontheotherhand,thevariables𝑤(ℓ)spotthe rizingthetotalanomalylevelofthequery.Yet,westressthat,for
𝑘
anomaliesnotcapturedby𝑠.Theydoitbycomparingtheactivity amorerefinedstudy,itispossibletorecomputeourmulti-scale
betweenE (ℓ) andE (ℓ),wheretheformerhasrelationsthatare analysisonthequeriesthatourapproachidentifiesasabnormal.
𝑘 𝑘+1 Weproduceanomalyscorefortheentirequeryasfollows:
moreprobabletoappearthanthelatter.Thisway,agroupoflikely
r o n e n a l t e a u s t r io i a n n lq s E u i 𝑘 ( n e ℓ + s ) 1 E ti s 𝑘 o ( u ℓ n ) dd t s h u e a n d t l d y m en a a p l y y p a e d r a i i s r s a i e n p i g p s e h w a a r h v i y n e g ( a 5 a ) s n t a r d c o t a n iv g g i r t i i o m e u s p p b a o c e t t f w i l n e e s e 𝑤 s n 𝑘 l ( i ℓ s k ) u e . c l A h y score(1 𝜙 𝜙ˆ )= (𝑠− 𝜎2 E [𝑠 [𝑠 ] ]) 2 + ∑︁ ℓ,𝑘 (cid:16) 𝑤 𝑘 (ℓ 𝜎 ) 2 − [𝑤 E 𝑘 [ ( 𝑤 ℓ) 𝑘 ( ] ℓ) ] (cid:17)2 . (8)
specificchoicesofsubsetsofrelationsarecomparedonly,given
thattherearemanymorewaysinwhichtwogroups,onewith Figure3providesacomprehensiveillustrationofourmulti-scale
elementsmoreprobablethantheother,maybeenvisagedtodefine approachtoanomalydetection.Inshort,MADtakesasinputa
similarrandomvariables.Ournextresultdemonstratesthatthe historyofpastinteractionsH andqueryQ.Itconstructsamodel𝑃
familydefinedby(4) and(5)alreadycontainsall thenecessary fromH (seeSection3.2)andusesittosettheorderingofthetree.
detailstodiscernanomalies,asitdoesnotinvolveanyinformation Then,thetreeisusedtodecomposethequeryintoasetofrandom
𝜙 variables.MADalsoleveragesthemodeltoestimatethetheoretical
lossabout1 .
𝜙ˆ momentsofthemulti-scalerandomvariables.Then,itmeasures
howmanystandarddeviationsawayfromthemeanthequeryisin
Proposition1. Let1 𝜙
𝜙ˆ
and{𝑠,𝑤
𝑘
(ℓ) }denoteabinarystatefunc-
ordertosetananomalyscore.WhileMADsetsequalimportanceto
tionanditsassociatedsetofrandomvariablesasdefinedin(4)and thedifferentrandomvariablesinvolvedinthecomputationofthe
(5),respectively.Itholdsthat: anomalyscore,makingbothlarge-scaleorsmall-scaleanomalies
√ equallyrelevant,westressthatitispossibletofavoranomaliesat
1 𝜙 = √ 1 𝑠1 𝜙 + ∑︁ √ 2ℓ 𝑤(ℓ) (cid:20) 1 𝜙 −1 𝜙 (cid:21) . (6) anydesiredscalebygivingmoreweighttotherandomvariables
𝜙ˆ 𝑀 𝜙 𝑀 𝑘 E(ℓ+1) E(ℓ+1) associatedtosuchlevel.
ℓ,𝑘 2𝑘 2𝑘+1
42

WSDM’24,March4–8,2024,Merida,Mexico EstebanBautista,LaurentBrisson,CécileBothorel,&GrégorySmits
Figure3:Schematicrepresentationoftheproposedmulti-scaleapproachforanomalydetection.
3.2 EstimationoftheModelProbabilities Synthetic Hospital Emails Traffic
Thissectionaddressestheestimationofthemodel𝑃assumedabove Triplets 1.24M 32.4K 30.7K 382K
fromthehistoryH.𝑃mustrepresentwhatweintendbynormality. Nodes 925 75 1646 1622
Inthiswork,weassumethatnormalinteractionsshouldbelocally Max.time 5K 17.4K 44.5K 7.2K
stationary.Stationaritymeansthattheunderlyingrandomprocess Emptyslices 0 7.9K 32K 0
generatingthedataremainsstableovertime.Thus,weassumethat Activitypeak 308 20 76 94
thereisasinglemodelthatproducedinteractionsintherecentpast
Table2:Datasetsstatistics.
andthat,inordertoconsidertheinteractionsattime𝑡 asnormal,
they should also be generated by such model. Our challenge is
thustospotthemodelthatproducedinteractionsintherecent
past and use it to define normality at time𝑡. Notice that if we twodistributionsdiffer.Thistestallowsustoautomaticallyexplore
identifyawindowinwhichtheinteractionsarestationary,thenwe the𝑁-lengthcontexttoidentifythesub-windowofsize𝐾inwhich
canstraightforwardlyestimate𝑃 throughasimpletimeaveraging. thestationarityassumptionbestholds:wesimplyrunawindow
Thisisbecausestationaritymeansthatalltheobservedstatesof backwardly,startingfrom𝑡−1,forallpossiblevaluesof𝐾andrun
relation𝑒 𝑖 overtimearesamplesofthesameBernoulliexperiment thetestforeachwindow,retainingtheoneinwhichthedistribu-
ofprobability𝑃(𝑒 𝑖).Hence,𝑃(𝑒 𝑖)canbeestimatedfromitstime tionsbestmatch.Thematchofdistributionsisquantifiedbysetting
samplesas: afitnessscoregivenbythesumofsquareddifferencesbetweenthe
𝐾
sampleandtheoreticalvarianceforeachrandomvariable.1
1 ∑︁ 𝜙
𝑃 = 1 (9)
𝐾 𝐿(𝑡−𝑘,𝜙)
𝑘=1 4 NUMERICALEXPERIMENTS
where𝐾 isthelengthofthestationarywindow.Thechallengeof ThissectionevaluatestheperformanceofMADthroughexperimen-
estimating𝑃thereforeliesinidentifyingasub-windowoflength𝐾 tationsthataimtoaddressthefollowingquestions.Q1.Accuracy:
fromthecontextoflength𝑁 inwhichalltheslicesarestationary. howaccuratelycanMADdetectanomalouseventsconsistingcom-
Notably,wecanleverageourmulti-scaleanalysistodesignasimple paredtobaselines?Q2.Flexibility:canMADhandleequallywell
stationaritytestthataddressesthischallenge. queriesofvaryingform?Q3.Interpretability:doesMADallow
Theideaofourstationaritytestisanhypothesistestingone:we tocharacterizethesignatureofabnormalevents?Theimplementa-
hypothesizethatthewindowisstationaryandthenwetrytoreject tionofMADandcodetoreproducetheexperimentsisavailablein
thehypothesisusingourmulti-scaleanalysis.Assumingstationary https://gitlab.imt-atlantique.fr/publications1/mad.
meansthatallthe𝐾 sliceswithinthewindowarerealizationsofa Datasets. Onesyntheticandthreerealworlddatasetsareused
same𝑃,whichcanbeestimatedasin(9).Then,ifwecomputeone (Table2).Thesyntheticoneiscomposedofagraphsequencewith
ofourrandomvariablesacrossallthesliceswithinthewindow,we stablecommunitystructurebutwheresomeedgesappearmore
mustobtain𝐾 valuesthataredistributedaspredictedbyProposi- frequentlythanothers.Itisdonebyfixingamodel𝑃andgenerating
tion2,giventhattheyarerealizationsofthesame𝑃.Onecanassess asequenceofrealizationsaccordingtotheproceduredetailedin
ifthese𝐾valuesareindeeddistributedinsuchawaybycomparing Section3.1.Themodel𝑃 consistsofaheterogeneousstochastic
theirsamplemomentswiththetheoreticalonespredictedbyPropo-
1Onlythevarianceisemployedasthefactthat𝑃isestimatedusingasamplemean
sition2.Thestationarityhypothesisisthereforerejectedifthese impliesthattheexpectationsfromProposition2equalthesampleones.
43

MAD:Multi-ScaleAnomalyDetectioninLinkStreams WSDM’24,March4–8,2024,Merida,Mexico
blockmodel:edgeswithinandbetweencommunitieshavedifferent MIDAS F-FADE MAD
probabilities.Seethesupplementaryforadetaileddescriptionof
|     |     |     |     |     | Synthetic | 0.49 | 0.53 0.58 |
| --- | --- | --- | --- | --- | --------- | ---- | --------- |
howthemodelisset.Realdatasetsare:Hospital[19]containing
|     |     |     |     |     | Hospital | 0.50 | 0.80 |
| --- | --- | --- | --- | --- | -------- | ---- | ---- |
0.82
| temporalinteractionsbetweenpatientsandhealth-careworkers |     |     |     | Densification |        |      |           |
| -------------------------------------------------------- | --- | --- | --- | ------------- | ------ | ---- | --------- |
|                                                          |     |     |     |               | Emails | 0.73 | 0.98 0.76 |
inahospitalward(slicesof20-seconds);Emails[20]:thedirected
|     |     |     |     |     | Traffic | 0.52 | 0.56 |
| --- | --- | --- | --- | --- | ------- | ---- | ---- |
network of emails in the 2016 Democratic National Committee 0.76
emailleak(slicesofoneminute);Traffic[21]:two-hoursofTCP Synthetic - -
0.80
trafficbetweentheLawrenceBerkeleyLaboratoryandtherestof
|                                                            |     |     |     |                | Hospital | -   | - 0.85 |
| ---------------------------------------------------------- | --- | --- | --- | -------------- | -------- | --- | ------ |
| theworld(slicesofonesecond).Ingeneral,thesedatasetsarevery |     |     |     | Sparsification |          |     |        |
|                                                            |     |     |     |                | Emails   | -   | - 0.84 |
| dynamicandsparse.                                          |     |     |     |                | Traffic  | -   | -      |
0.89
| Anomalyinjection. | Therearenoknownanomalieswithin |     |     |     |     |     |     |
| ----------------- | ------------------------------ | --- | --- | --- | --- | --- | --- |
theselecteddatasets.Therefore,thewholedatasetsareconsiderd Table3:EdgeanomalydetectionperformanceinAUC.
normalanddifferentabnormaleventsareadded:(i)suddenden-
sifications;(ii)suddensparsifications;and(iii)suddenrewirings.
Weinjectabnormaleventsaccordingtothetypeofqueriestobe F-FADE-N∗ DynAnom MAD
assessed(seethesupplementaryforafulldescription):
|     |     |     |     |     | Synthetic | 0.52 | 0.56 |
| --- | --- | --- | --- | --- | --------- | ---- | ---- |
0.88
| • Edgeanomalies.                                     | Arelation(𝑢,𝑣)israndomlyselectedand |     |     | Densification |          |      |           |
| ---------------------------------------------------- | ----------------------------------- | --- | --- | ------------- | -------- | ---- | --------- |
|                                                      |                                     |     |     |               | Hospital | 0.82 | 0.51 0.92 |
| attackedatvarioustimes.Densificationattacksmake(𝑢,𝑣) |                                     |     |     | &             |          |      |           |
|                                                      |                                     |     |     |               | Emails   | 0.82 | 0.54      |
active at times where it is very infrequent, while sparsi- 0.84
Sparsification
fication attacks suppress (𝑢,𝑣) at times where it appears Traffic 0.77 0.51 0.74
frequently.Foreachdataset,50relationsareattacked.
|     |     |     |     |     | Synthetic | 0.53 | 0.52 0.83 |
| --- | --- | --- | --- | --- | --------- | ---- | --------- |
• Nodeanomalies.Arandomlyselectednodeisattackedatvar- Hospital 0.57 0.54 0.99
Rewiring
ioustimeswithdensifications/sparsificationsorrewirings. Emails 0.59 0.51
0.99
Theformerattackinjects/suppressescommunicationsemerg-
|          |                    |                  |               |     | Traffic | 0.53 | 0.54 0.99 |
| -------- | ------------------ | ---------------- | ------------- | --- | ------- | ---- | --------- |
| ing from | the attacked node, | while the latter | redirects its |     |         |      |           |
communicationstowardsothernodes.Creatededgesdueto Table4:NodeanomalydetectionperformanceinAUC.∗F-
FADE-Nisevaluatedonlyonthesubsetofscoresthatitis
attacksalwayspointtowardsnodesthattheattackednode
abletoproduce.
hasalreadycommunicatedwith.Foreachdataset,10nodes
areattackedovertime.Attacksareboundedto3edges.
| •                | Anomalies | here concern | densification/- |     |     |     |     |
| ---------------- | --------- | ------------ | --------------- | --- | --- | --- | --- |
| Graph anomalies. |           |              |                 |     |     |     |     |
sparsificationorrewiringeventsappliedtolinkstreamslices
tothefactthat(i)MIDASconsidersglobalaggregatesandhenceis
chosenatrandom.Foreachdataset,1%ofitsactiveslices
agnostictoshortintervalsoflowactivity;and(ii)F-FADErequires
areattacked.Attacksareboundedto5edges.
astableembeddingtoproduceaccuratefrequencyestimationsand
Baselines. Sixstate-of-the-artalgorithmsformthebaselines. isonlyabletoattainitforthedatasetsthathavemanyemptyslices.
Twoforedgeanomalies:MIDAS[1]andF-FADE[2].Twofornode
Moreover,westressthatMIDASandF-FADEcannotrespondto
anomalies:DynAnom[6]andF-FADE-N[2],thevariantofF-FADE
queriesconsistingofinactiverelations,makingthemunabletospot
proposedbytheirauthorstoaddressnodeanomalies.Twoforgraph
sparsifications.NoticethatMADsolvesthesetwoissuesbybeing
anomalies:AnomRank[8]andLAD[10].
abletospotthetwoanomalytypesandinaconsistentmanner.
Additionally,MADdoesitbyjustconsideringacontextbasedon
4.1 AccuracyofMAD
thepastactivityofthequeryedgewhileF-FADEneedstouseall
ThissubsectionaimstoaddressQ1andQ2byassessingtheaccuracy thepastlinkstreaminteractions.
ofMADandbaselines,inAUCscore,inthetasksofedge,nodeand Nodedetection.MADandbaselinesarealsoevaluatedina
graphanomalydetection.Forallexperimentswetriednumerous nodedetectionsettingtroughasimilarexperimentalsetup:algo-
hyper-parametersconfigurationsandretainedthebestones.See rithmsareaskedtodeterminetheabnormalityofeachnode𝑢that
thesupplementaryforourchoicesofhyper-parameters. wasattackedforallpossibletimestamps.F-FADE-Nisnotableto
Edgedetection.MADandedge-anomalybaselinesareques- produceananswerforquerieswithnocommunicationsinthem,
tionedasfollows:foreachrelation(𝑢,𝑣)thatwasattacked,weask hence its accuracy is assessed on the subset of scores that it is
algorithmstoproduceananomalyscorefor(𝑢,𝑣)atallpossible abletoproduce.Twoversionsofeachdatasetareanalyzed,one
timestamps(attackedandnon-attackedtimes).Algorithmsmust withinjecteddensifications/sparsificationsandonewithrewiring
| thenreturnhighscoresfortimestampsatwhich(𝑢,𝑣)wasattacked. |     |     |     | events. |     |     |     |
| --------------------------------------------------------- | --- | --- | --- | ------- | --- | --- | --- |
Twoversionsofeachdatasetareanalyzed,onewithinjecteddensi- Table4clearlyshowsthatMADperformsverywellinthede-
ficationsandonewithinjectedsparsifications. tectionofbothsparsification/densificationandrewiringevents.In
ResultsarereportedinTable3.ItcanbeseenthatMADsystemat- particular,MADisabletodetecttherewiringeventsinthereal
icallyperformswellindetectingbothanomalies,whileMIDASand datasetswithalmostperfectaccuracy.SuchaperformanceofMAD
F-FADEareinconsistentwithdensificationsandtheycannothandle isduetothefactthatrewiringeventsinthoseverysparsedatasets
sparsificationanomalies.Suchinconsistentbehaviormaybedue essentiallyreplacetheirfewlikelyedgeswithunlikelyonesonly,
44

WSDM’24,March4–8,2024,Merida,Mexico EstebanBautista,LaurentBrisson,CécileBothorel,&GrégorySmits
makingtheattackedqueriesextremelyinconsistentwiththerecent AnomRank LAD MAD
past.ItcanbeobservedthatF-FADE-Nperformswellindetect-
Synthetic 0.49 0.50 0.76
ingdensifications/sparsifications,eventhoughitonlyproducesan Densification Hospital 0.51 0.80 0.95
outputattimeswhenthequeriednodeshavecommunicationsin &
them.Thus,amassiveeventthatcompletelyshuts-downanode Sparsification Emails 0.54 0.92 0.98
wouldbemissedbyF-FADE-N.DynAnomsystematicallyperforms Traffic 0.43 0.46 0.77
verypoorly.Thispoorperformanceshouldnotcomeasasurprise: Synthetic 0.59 0.52 0.63
DynAnomcomputesanomalyscoresbasedonthestabilityofthe Hospital 0.58 0.77 0.94
Rewiring
PageRankofnodes,whichisclearlynotameaningfulfeaturefor Emails 0.53 0.87 0.87
suchsparseandquicklyevolvinglinkstreams.
Traffic 0.44 0.55 0.85
Graphdetection.TheaccuracyofMADandbaselinesareeval-
uatedinagraphdetectionsettingbyfeedingthealgorithmswith Table5:GraphanomalydetectionperformanceinAUC.
theslicesoftheattackeddatasets.Twoversionsofeachdatasetare
analyzed,onewithinjecteddensifications/sparsificationsandone
withrewiringevents.
ResultsarereportedinTable5.Asitcanbeseen,MADperforms
verywellinthedetectionofbotheventsregardlessofthedataset,
whileLADperformsinconsistentlyandAnomRankverypoorly.As
inthenodecase,attacksmakelikelyedgesdisappearandunlikely
onesappear.Therefore,MADseesthosecollectiveeventsashardto
explainfromthecontext,explainingitsgoodaccuracy.LADisin-
consistentasitdependsonthestabilityofeigenvaluedistributions,
whichisnotguaranteedwhenedgesarefullyreplacedbetween
snapshots.AnomRankreliesonPageRank,thusitsuffersfromthe
sameissuesofDynAnom.
4.2 InterpretabilityofMAD
Inthissubsection,Q3isaddressedbystudyinghowthedifferent Figure4:Distributionofanomalyscoresacrossrandomvari-
attacksinfluencetheindividualanomalyscoresproducedbyEqua- ables.Differentattacksproducedifferentsignatures.
tion(7).Thisstudyisconductedby(i)takingourmodel𝑃 usedto
generatesyntheticdata;(ii)generatinganormalgraphusingthe
thattheanomalyscoresofthiseventcombinethesignatureofthe
model;(iii)applyingdifferenttypesofattacksonthisgraph;(iv)
previoustwo.Thus,insum,thedifferentattacksproducediffer-
performingthemulti-scaleanalysistoeachoftheresultinggraphs;
entsignaturesinouranomalyscores,pavingthewaytostudythe
and(v)computingtheanomalyscoresofeachrandomvariable.
signatureofmorecomplexandrealeventsasfurtherresearch.
Figure4displaysthedistributionofanomalyscoresforthedif-
ferenttypesofattacks.Therandomvariablesareorderedsothat
theleft-mostonesintheplotaretheonesassociatedwiththecoars- 5 CONCLUSION
estscales,i.e.𝑠 and𝑤(0),andtheright-mostonesaretheones InthisworkweintroducedMAD,aMulti-scaleAnomalyDetection
0
associatedtothefinestscales.Asitcanbeseen,thenormalgraph algorithmforlinkstreamsthatallowstoevaluateifanyarbitrary
produceslowanomalyscoresformostrandomvariables:onlyfew time-stampedsubgraphisabnormal.Throughanumericaleval-
fine-scaleoneshavelargescores,whichisduetotheinherentun- uation,wedemonstratedthatMADperformssignificantlybetter
certaintyassociatedtoagraphgeneratedatrandom.Whenthis thanstate-of-the-artalternatives,evenwhenthedataathandis
graphissubjecttoadensificationattack,itcanbeseenthatalarge veryuncertainandsparse,inthetasksofdetectingedges,nodesor
numberofrandomvariablesimmediatelyyieldlargescores.Since graphsthatweresubjecttodensification,sparsificationandredirec-
thelikelyedgesremainpresentinthegraphandthemajorityofun- tionattacks.ThisflexibilityandgoodaccuracyofMADstemsfrom
likelyonesremainsinactiveintheattackedgraph,thelargescores itsscoringmechanism,whichbuildsonanovelprobabilisticand
mostlyappearatfinescalesasmostoftheactivityinthegraph multi-scaleanalysisofsub-graphsthatallowstodecomposethem
remainswellexplainedbythemodel.Yet,noticethat𝑠immediately intoasetofrandomvariablesthatcaptureanomaliesatvarious
activates,thuspointingthedensification.Noticethatasparsifi- resolutionscales.ThismakesMADnotonlyaccuratebutalsoinher-
cationattacksuppressesmostlikelyedgesandthisimmediately entlyinterpretableandtheoreticallysound.Thenextstepconcerns
triggersthescoresassociatedtocoarseresolutions,particularly𝑠 thecombinationofMADwithananomalyexplanationmechanism
and𝑤(0).Therandomvariables𝑠and𝑤(0) significantlyincrease toassistfinalusersintheanalysisofthefoundanomalies.
0 0
becausetheactivityofthisgraphdoesnotmatchtheexpectedone
fortheformer,andbecausetheattackmostlyaffectstheleft-side ACKNOWLEDGEMENTS
ofthetreeforthelatter.Sincearewiringattackisessentiallya TheauthorswouldliketothanktheCarnotTélécom&Société
combinationofasparsificationandadensification,onecanremark Numériqueinstituteforthefinancialsupport.
45

MAD:Multi-ScaleAnomalyDetectioninLinkStreams WSDM’24,March4–8,2024,Merida,Mexico
REFERENCES [11] M.McNeil,C.Mattsson,F.W.Takes,andP.Bogdanov,“Cadence:Community-
[1] S.Bhatia,B.Hooi,M.Yoon,K.Shin,andC.Faloutsos,“Midas:Microcluster-based awaredetectionofdynamicnetworkstates,”inProceedingsofthe2023SIAM
detectorofanomaliesinedgestreams,”inProceedingsoftheAAAIconferenceon InternationalConferenceonDataMining(SDM),pp.1–9,SIAM,2023.
artificialintelligence,vol.34,pp.3242–3249,2020. [12] S.Fernandes,H.Fanaee-T,J.Gama,L.Tišljarić,andT.Šmuc,“Wintended:Win-
[2] Y.-Y.Chang,P.Li,R.Sosic,M.Afifi,M.Schweighauser,andJ.Leskovec,“F-fade: dowedtensordecompositionfordensificationeventdetectionintime-evolving
Frequencyfactorizationforanomalydetectioninedgestreams,”inProceedingsof networks,”MachineLearning,vol.112,no.2,pp.459–481,2023.
the14thACMInternationalConferenceonWebSearchandDataMining,pp.589– [13] Y.JiangandG.Liu,“Two-stageanomalydetectionalgorithmviadynamiccommu-
597,2021. nityevolutionintemporalgraph,”AppliedIntelligence,vol.52,no.11,pp.12222–
[3] D.EswaranandC.Faloutsos,“Sedanspot:Detectinganomaliesinedgestreams,” 12240,2022.
in2018IEEEInternationalconferenceondatamining(ICDM),pp.953–958,IEEE, [14] Z.TasnádiandN.Gaskó,“Anewtypeofanomalydetectionproblemindynamic
2018. graphs:Anantcolonyoptimizationapproach,”inInternationalConferenceon
[4] R.PaudelandH.H.Huang,“Pikachu:Temporalwalkbaseddynamicgraph BioinspiredOptimizationMethodsandTheirApplications,pp.46–53,Springer,
embeddingfornetworkanomalydetection,”inNOMS2022-2022IEEE/IFIPNetwork 2022.
OperationsandManagementSymposium,pp.1–7,IEEE,2022. [15] M.BansalandD.Sharma,“Density-basedstructuralembeddingforanomaly
[5] L.Fang,K.Feng,J.Gui,S.Feng,andA.Hu,“Anonymousedgerepresentation detectionindynamicnetworks,”Neurocomputing,vol.500,pp.724–740,2022.
forinductiveanomalydetectionindynamicbipartitegraph,”Proceedingsofthe [16] P.Jiao,T.Li,Y.Xie,Y.Wang,W.Wang,D.He,andH.Wu,“Generativeevolutionary
VLDBEndowment,vol.16,no.5,pp.1154–1167,2023. anomalydetectionindynamicnetworks,”IEEETransactionsonKnowledgeand
[6] X.Guo,B.Zhou,andS.Skiena,“Subsetnodeanomalytrackingoverlargedy- DataEngineering,2021.
namicgraphs,”inProceedingsofthe28thACMSIGKDDConferenceonKnowledge [17] A.Wilmet,T.Viard,M.Latapy,andR.Lamarche-Perrin,“Degree-basedoutliers
DiscoveryandDataMining,pp.475–485,2022. detectionwithiniptrafficmodelledasalinkstream,”in2018NetworkTraffic
[7] N.A.Heard,D.J.Weston,K.Platanioti,andD.J.Hand,“Bayesiananomaly MeasurementandAnalysisConference(TMA),pp.1–8,IEEE,2018.
detectionmethodsforsocialnetworks,”2010. [18] C.C.Aggarwal,Y.Zhao,andS.Y.Philip,“Outlierdetectioningraphstreams,”in
[8] M.Yoon,B.Hooi,K.Shin,andC.Faloutsos,“Fastandaccurateanomalydetection 2011IEEE27thinternationalconferenceondataengineering,pp.399–409,IEEE,
indynamicgraphswithatwo-prongedapproach,”inProceedingsofthe25th 2011.
ACMSIGKDDInternationalConferenceonKnowledgeDiscovery&DataMining, [19] P.Vanhems,A.Barrat,C.Cattuto,J.-F.Pinton,N.Khanafer,C.Régis,B.-a.Kim,
pp.647–657,2019. B.Comte,andN.Voirin,“Estimatingpotentialinfectiontransmissionroutesin
[9] D.Eswaran,C.Faloutsos,S.Guha,andN.Mishra,“Spotlight:Detectinganom- hospitalwardsusingwearableproximitysensors,”PloSone,vol.8,no.9,p.e73970,
aliesinstreaminggraphs,”inProceedingsofthe24thACMSIGKDDInternational 2013.
ConferenceonKnowledgeDiscovery&DataMining,pp.1378–1386,2018. [20] J.Kunegis,“KONECT–TheKoblenzNetworkCollection,”inProc.Int.Conf.on
[10] S.Huang,Y.Hitti,G.Rabusseau,andR.Rabbany,“Laplacianchangepointdetec- WorldWideWebCompanion,pp.1343–1350,2013.
tionfordynamicgraphs,”inProceedingsofthe26thACMSIGKDDInternational [21] V.PaxsonandS.Floyd,“Wideareatraffic:thefailureofpoissonmodeling,”
ConferenceonKnowledgeDiscovery&DataMining,pp.349–358,2020. IEEE/ACMTransactionsonnetworking,vol.3,no.3,pp.226–244,1995.
46
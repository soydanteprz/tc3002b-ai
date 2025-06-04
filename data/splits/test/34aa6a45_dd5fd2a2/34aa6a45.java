import java.util.*;
import java.math.*;
import java.io.*; 
 public class B{
static int[] dx = new int[]{-1,0,1,0};
    static int[] dy = new int[]{0,1,0,-1};
    static FastReader scan=new FastReader();
   public static PrintWriter out = new PrintWriter (new BufferedOutputStream(System.out));
  static ArrayList<Pair>es;
  static LinkedList<Integer>edges[];
  static Set<Pair>edges2[][];

   static boolean prime[];
         
     
    
    
      

  
   






    
      
  
    
   
       
    
   static int mid;
   

     
 
   



static int cnt=0;
static int count1[];
  
/*static int rec(int i)
{

}*/
static long arr[];
static int n;
static long res,tmpres;
static ArrayList<Pair>pairs;
static boolean vis[];
static long dp[][];
static long dfs(int x,long last,int prev)
{
  //  vis[x]=true;
   // System.out.println
    if(dp[x][prev]!=-1){
       // out.println("FUCK");
        return dp[x][prev];
    }
    long ch1=0,ch2=0,res=0;
    for(int v:edges[x])
    {
        if(!vis[v])
        {
            vis[v]=true;
          //  ch1+=

            //vis[v]=false;
            res+=Math.max((dfs(v,pairs.get(v).x,1)+Math.abs(pairs.get(v).x-last)),(dfs(v,pairs.get(v).y,2)+Math.abs(pairs.get(v).y-last)));
           // res+=Math.max(ch1,ch2);
            vis[v]=false;
        }
    }
    return dp[x][prev]=res;
}
public static void main(String[] args)  throws Exception 
{
    /*int xx=253;
    for(int i=1;i*i<=xx;i++)
    {
        if(xx%i==0)
        {
            System.out.println(i);
            System.out.println(xx/i);
        }
    }*/
//java.util.Scanner scan=new java.util.Scanner(new File("mootube.in"));
// PrintWriter out = new PrintWriter (new FileWriter("mootube.out"));
//scan=new FastReader("skidesign.in");
//out = new PrintWriter ("skidesign.out");
//System.out.println(3^2);
//System.out.println(19%4);
   //StringBuilder news=new StringBuilder("ab");
   //news.deleteCharAt(1);
   //news.insert(0,'c');
   //news.deleteCharAt(0);
   //System.out.println(news);
   //System.out.println(can(2,15));
    //System.out.println(LCM(2,2));
  // System.out.println(31^15);
 //System.out.println("");
   //System.out.println(824924296372176000L>(long)1e16);
int tt=1;
//rec(2020);
//System.out.println(Long.MAX_VALUE);
//System.out.println(Integer.MAX_VALUE);
//int st=scan.nextInt();
//System.out.println(calc(91));
//sieve(21000);

//SNWNSENSNNSWNNW

   // System.out.println(set.remove(new Pair(1,1)));
//System.out.println(count("cccccccccccccccccccccccccooooooooooooooooooooooooodddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeffffffffffffforrrrrrrrrrrrrcesssssssssssss","codeforces"));
  //S0ystem.out.println(isPowerOfTwo(446265625L));
//System.out.println("daaa".compareTo("bccc"));
//System.out.println(2999000999L>1999982505L);
//System.out.println("?bac?bac??a?a?a?abac????abacab?ca?a".length());
//StringBuilder s=new StringBuilder("");
//tt=scan.nextInt();
//String ok="ABDC";
//System.out.println(ok.substring(2));
//System.out.println(2000000000/2);
tt=scan.nextInt();
//System.out.println(LCM(4,6));
//System.out.println(isprime(23335));
//int T=1;
  outer:while(tt-->0)
 {
int n=scan.nextInt();
pairs=new ArrayList<Pair>();
edges=new LinkedList[n];
for(int i=0;i<n;i++)
edges[i]=new LinkedList();
for(int i=0;i<n;i++)
{
    long l=scan.nextLong(),r=scan.nextLong();
    pairs.add(new Pair(l,r));
}
for(int i=0;i<n-1;i++)
{
    int a=scan.nextInt()-1;
    int b=scan.nextInt()-1;
    edges[a].add(b);
    edges[b].add(a);
}
vis=new boolean[n];
vis[0]=true;
dp=new long[n][3];
for(long K[]:dp)
    Arrays.fill(K,-1);
long x=dfs(0,pairs.get(0).x,0);
vis=new boolean[n];
vis[0]=true;
dp=new long[n][3];
for(long K[]:dp)
    Arrays.fill(K,-1);
out.println(Math.max(x,dfs(0,pairs.get(0).y,0)));

}
out.close();

}





 


 
public static class Pair implements Comparable<Pair>{
         long x;
         long y;
         long ab;
        
        long z;
        public Pair(){}
        public Pair(long x1, long y1,long z) {
            x=x1;
            y=y1;
            this.z=z;
        }
        public Pair(long x1, long y1) {
            x=x1;
            y=y1;
            this.ab=x+y;
        
        }
        
        
        
        
}

}

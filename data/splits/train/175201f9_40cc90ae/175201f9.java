import java.util.*;
import java.io.*;
public class Sol{
 
/*   
->check n=1, int overflow , array bounds , all possibilites(dont stuck on 1 approach)
->Problem = Observation(constraints(m<=n/3 or k<=min(100,n)) 
           + Thinking + Technique (seg_tree,binary lift,rmq,bipart,dp,connected comp etc)
->solve or leave it (- tutorial improves you in minimal way -)
*/
 
public static void main (String []args) {
//precomp();
 
int times=ni();while(times-->0){solve();}out.close();}

static void solve(){
 int n=ni();
 int sum[]=new int[5]; 
 PriorityQueue<Integer> p[]=new PriorityQueue[5];for(int i=0;i<5;i++)p[i]=new PriorityQueue<Integer>();
 for(int i=0;i<n;i++){
   char c[]=ns().toCharArray();
   int len=c.length;
   int csum[]=new int[5];
   for(int j=0;j<len;j++){csum[c[j]-'a']++;}
   for(int j=0;j<5;j++){csum[j]=2*csum[j]-len;p[j].offer(csum[j]);sum[j]+=csum[j];}
 }
 int max=0;
 for(int i=0;i<5;i++){
   int cmax=n;
   while(p[i].size()>0 && sum[i]<=0){cmax--;sum[i]-=p[i].poll();}
   max=Math.max(max,cmax);
 }
 out.println(max);
 return;

}

//-----------------Utility--------------------------------------------


 
 

 


 
//----------------------I/O---------------------------------------------
 
static InputStream inputStream = System.in;
static OutputStream outputStream = System.out;
static FastReader in=new FastReader(inputStream);
static PrintWriter out=new PrintWriter(outputStream);
 

/*static int ni() {
        try {
            boolean in = false;
            int res = 0;
            for (;;) {
                int b = System.in.read() - '0';
                if (b >= 0) {
                    in = true;
                    res = 10 * res + b;
                }
                else if (in)
                    return res;
            }
        } catch (IOException e) {
            throw new Error(e);
        }
    }*/
static int ni(){return in.nextInt();}


static String ns(){return in.nextLine();}
}

import java.io.*;
import java.util.*;
import java.math.BigInteger;


public class C{
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Task solver = new Task();
        solver.solve(in, out);
        out.close();
    }
    // main solver
    static class Task{

        double eps= 0.00000001;
        static final int MAXN = 100005;
        static final int MOD= 998244353;

        // stores smallest prime factor for every number
        static int spf[] = new int[MAXN];

        Map<Integer,Set<Integer>> dp= new HashMap<>();

        // Calculating SPF (Smallest Prime Factor) for every
        // number till MAXN.
        // Time Complexity : O(nloglogn)
        

        // A O(log n) function returning primefactorization
        // by dividing by smallest prime factor at every step
        
        // function to find first index >= x
        

        // function to find last index <= y
        

        // function to count elements within given range
        

        

        

        

        int[] fact= new int[MAXN];
        int[] inv= new int[MAXN];

        

        

        
        
        
        public void solve(InputReader in, PrintWriter out) {
            int test= in.nextInt();
            while(test-->0){
                int n= in.nextInt(), k= in.nextInt();
                int[] a= new int[k];
                int[] t= new int[k];
                for(int i=0;i<k;i++) a[i]= in.nextInt();
                for(int i=0;i<k;i++) t[i]= in.nextInt();
                int[] c= new int[n];
                Arrays.fill(c, Integer.MAX_VALUE);
                for(int i=0;i<k;i++){
                    c[a[i]-1]=t[i];
                }
                int[] L= new int[n];
                int[] R= new int[n];
                Arrays.fill(L, Integer.MAX_VALUE);
                Arrays.fill(R, Integer.MAX_VALUE);

                long p= Integer.MAX_VALUE;
                for(int i=0;i<n;i++){
                    p=Math.min(p+1,c[i]);
                    L[i]=(int)p;
                }
                p= Integer.MAX_VALUE;
                for(int i=n-1;i>=0;i--){
                    p= Math.min(p+1,c[i]);
                    R[i]=(int)p;
                }
                for(int i=0;i<n;i++){
                    out.print(Math.min(L[i],R[i])+" ");
                }
                out.println();
            }
            
        }
    

        // public static class compareL implements Comparator<Tuple>{
        //     @Override
        //     public int compare(Tuple t1, Tuple t2) {
        //         return t2.l - t1.l;
        //     }
        // }
        
        

    }

    


    

    
    // fast input reader class;
    
}

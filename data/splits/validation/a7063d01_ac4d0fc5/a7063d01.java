import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    public static void main(String[] args){
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        solve(in, out);
        out.close();
    }

    

    

    

    

    

    

    

    

    static int parent[];
    static int rank[];

    static long m = 0;

    
    
    
    
    

    static int parent1[];
    static int rank1[];

    
    
    
    
    

    static long max1 = 0;

    static int count = 0;
    static int count1 = 0;

    static boolean possible;

    public static void solve(InputReader sc, PrintWriter pw){
        int i, j = 0;
        // int t = 1; 
        long mod = 1000000007;
        // int factors[] = new int[1000001];
        // ArrayList<Integer> ar = new ArrayList<>();
        // sieveOfEratosthenes(1000000, factors, ar);
        // HashSet<Integer> set = new HashSet<>();
        // for(int x:ar){
        //     set.add(x);
        // }
        int t = sc.nextInt();
        u: while (t-- > 0) {
            int n = sc.nextInt();
            int e[][] = new int[n-1][2];
            int x[] = new int[n];
            int m = 0;
            for(i=0;i<n-1;i++){
                e[i][0] = sc.nextInt()-1;
                e[i][1] = sc.nextInt()-1;
                x[e[i][0]]++;
                x[e[i][1]]++;
                m = Math.max(x[e[i][0]],m);
                m = Math.max(x[e[i][1]],m);
            }
            if(m>2)
                pw.println(-1);
            else{
                if(n==2){
                    pw.println(2);
                }
                else if(n==3){
                    pw.println(2+" "+3);
                }
                else{
                    int d = 0;
                    int ans[] = new int[n-1];
                    ArrayList<Integer> ar[] = new ArrayList[n];
                    ArrayList<Integer> ar1[] = new ArrayList[n];
                    for(i=0;i<n;i++){
                        ar[i] = new ArrayList<>();
                        ar1[i] = new ArrayList<>();
                    }
                    for(i=0;i<n-1;i++){
                        int a = e[i][0];
                        int b = e[i][1];
                        ar[a].add(b);
                        ar1[a].add(i);
                        ar[b].add(a);
                        ar1[b].add(i);
                        if(x[a]==1)
                            d = a;
                        if(x[b]==1)
                            d = b;
                    }
                    visit(d,ar,ar1,ans,-1,2);
                    for(i=0;i<n-1;i++){
                        pw.print(ans[i]+" ");
                    }
                    pw.println();
                }
            }
        }
    }

    static void visit(int d, ArrayList<Integer> ar[], ArrayList<Integer> ar1[], int ans[], int par, int v){
        if(ar[d].get(0)!=par){
            ans[ar1[d].get(0)] = v;
            visit(ar[d].get(0), ar, ar1, ans, d, 5-v);
            return;
        }
        if(ar[d].size()==1)
            return;
        ans[ar1[d].get(1)] = v;
        visit(ar[d].get(1), ar, ar1, ans, d, 5-v);
    }

    
  
    

    

    

    

    

    

    

    

    

    

    

    

     

    

    

    

    
}

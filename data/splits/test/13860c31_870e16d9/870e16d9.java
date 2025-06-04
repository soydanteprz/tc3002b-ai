import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class CF1 {
    public static void main(String[] args) {
        FastScanner sc=new FastScanner();
        int T=sc.nextInt();
//        int T=1;
        for (int tt=0; tt<T; tt++){
           int n =sc.nextInt();
           String a = sc.next();
           String b= sc.next();
           int candles1=0;
           int candles2=0;
           int inPlace=0;
           for (int i=0; i<n; i++){
               if (a.charAt(i)=='1'){
                   candles1++;
               }
               if (b.charAt(i)=='1') candles2++;
               if (b.charAt(i)=='1' && a.charAt(i)=='1') inPlace++;
           }
           if (candles1==candles2 || n+1-candles2==candles1){
             int ans=Math.min(help(candles1-inPlace, inPlace,n+1,candles2,0,0), help(candles1-inPlace, inPlace,n+1,candles2,0,1));
             if (ans==Integer.MAX_VALUE) System.out.println(-1);
             else System.out.println(ans);
           }
           else System.out.println(-1);
        }
    }
    static int help(int i, int j, int max, int cands, int moves, int flip){
        if (j==cands && i==0) return moves;
        else if (j==0 && flip==0 || flip==1 && i==0 ) return Integer.MAX_VALUE;

            if (flip==0){
                int x=0;
                int y=cands-j+1;
                if (i+j==max-cands){
                    x=cands-y;
                }
                else x=max-cands-y;
                return help(x,y,max,cands,moves+1,1);
            }
            else {
                int x=0;
                int y=cands-j;
                if (i+j==max-cands){
                    x=cands-y;
                }
                else x=max-cands-y;
                return help(x,y,max,cands,moves+1,0);
            }
        
    }
    static long mod =998244353L;
    

    
    
    
    

    
    
}

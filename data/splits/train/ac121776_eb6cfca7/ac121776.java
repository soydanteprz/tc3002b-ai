import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;


public class Solution {
    
    
    
    
    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        int t = sc.nextInt();
        while (t-- != 0) {
            int n = sc.nextInt();
            long a[] = new long[n];
            for(int i=0;i<n;i++){
                a[i] = sc.nextLong();
            }
            long mineA = a[0];
            long mineB = a[1];
            long accA =0 ;
            long accB = 0;
            long remA = n;
            long remB = n;
            long ans = (mineA * n) + (mineB *n);
            boolean check = true;
            for(int i =2;i<n;i++){
                if(check){
                    check = false;
                    long x = Math.min(a[i],mineA);
                    long y = Math.max(a[i],mineA);
                    mineA = x;
                    accA += y;
                    remA --;
                    long sug = (accA +  remA * mineA) + (accB + remB * mineB);
                    ans = Math.min(sug , ans);
                }else{
                    check = true;
                    long x = Math.min(a[i],mineB);
                    long y = Math.max(a[i],mineB);
                    mineB = x;
                    accB += y;
                    remB --;
                    long sug = (accA +  remA * mineA) + (accB + remB * mineB);
                    ans = Math.min(sug,ans);
                }
            }
            System.out.println(ans);
        }
    }
}

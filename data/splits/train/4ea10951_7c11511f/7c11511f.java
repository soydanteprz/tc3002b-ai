import java.io.*;
import java.util.*;

public class Solution {

    public static void main(String[] args) throws Exception {
        int tc = io.nextInt();
        for (int i = 0; i < tc; i++) {
            solve();
        }
        io.close();
    }

    private static void solve() throws Exception {
        int n = io.nextInt();
        String a = io.next();
        String b = io.next();
        int zz = 0;
        int zo = 0;
        int oo = 0;
        int oz = 0;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) == '1' && b.charAt(i) == '0') {
                oz++;
            }
            if (a.charAt(i) == '0' && b.charAt(i) == '0') {
                zz++;
            }
            if (a.charAt(i) == '1' && b.charAt(i) == '1') {
                oo++;
            }
            if (a.charAt(i) == '0' && b.charAt(i) == '1') {
                zo++;
            }
        }
        int ans = Integer.MAX_VALUE;
        if (oz == zo) {
            ans = Math.min(ans, oz + zo);
        }
        if (oo - 1 == zz) {
            ans = Math.min(ans, oo + zz);
        }
        if (ans == Integer.MAX_VALUE) {
            ans = -1;
        }
        io.println(ans);
    }

    

    //-----------PrintWriter for faster output---------------------------------
    public static FastIO io = new FastIO();

    //-----------MyScanner class for faster input----------
    
    //--------------------------------------------------------
}

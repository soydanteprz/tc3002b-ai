import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int t = scan.nextInt();
        for (int i = 0; i < t; i++) {
            int n = scan.nextInt();
            String a = scan.next();
            String b = scan.next();
            int[] cChet = new int[n];
            int[] cNech = new int[n];
            int sumChet = 0;
            int sumNechet = 0;
            int numOnesA = 0;
            int numOnesB = 0;
            boolean soluble = true;
            for (int j = 0; j < n; j++) {
                cChet[j] = (a.charAt(j) - 48) ^ (b.charAt(j) - 48);
                cNech[j] = cChet[j] ^ 1;
                sumChet += cChet[j];
                sumNechet += cNech[j];
            }
            int INF = 100000000;
            int ans = INF;
            if (((sumChet % 2 == 1) && (sumNechet % 2 == 0)) ||
                    !(numOnesB == numOnesA || numOnesB == (n - numOnesA + 1))) {
                ans = -1;
            } else {
                if (sumChet % 2 == 0) {
                    int ones = 0;
                    int zeroes = 0;
                    for (int j = 0; j < n; j++) {
                        if (cChet[j] == 1) {
                            if (a.charAt(j) == '1') {
                                ones++;
                            } else {
                                zeroes++;
                            }
                        }
                    }
                    if ((ones - zeroes) == 0 || (ones - zeroes) == 1) {
                        ans = Math.min(sumChet, ans);
                    }
                }
                if (sumNechet % 2 == 1) {
                    int ones = 0;
                    int zeroes = 0;
                    for (int j = 0; j < n; j++) {
                        if (cNech[j] == 1) {
                            if (a.charAt(j) == '1') {
                                ones++;
                            } else {
                                zeroes++;
                            }
                        }
                    }
                    if ((ones - zeroes) == 0 || (ones - zeroes) == 1) {
                        ans = Math.min(sumNechet, ans);
                    }
                }
            }
            if (ans != INF) {
                System.out.println(ans);
            } else {
                System.out.println(-1);
            }
        }
    }
    
}

import java.util.*;

public class mentor1 {
    public static boolean solve(int n, String color, int[] arr){
        List<Integer> Barr = new ArrayList<Integer>();
        List<Integer> Rarr = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) {
            if(color.charAt(i) == 'B')Barr.add(arr[i]);
            else Rarr.add(arr[i]);
        }
        Barr.sort(Comparator.naturalOrder());
        Rarr.sort(Comparator.reverseOrder());
        for (int i = 0; i < Barr.size(); i++) {
            if(Barr.get(i)< i + 1)return false;
        }
        for (int i = 0; i < Rarr.size(); i++) {
            int expect = n-i;
            if(Rarr.get(i) > expect)return false;
        }
        return true;
    }
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        int n = input.nextInt();
        for (int i = 0; i < n; i++) {
            int m = input.nextInt();
            int[] arr = new int[m];
            for(int j = 0;j<m; j++)arr[j] = input.nextInt();
            String color = input.next();
            if(solve(m,color,arr)) System.out.println("YES");
            else System.out.println("NO");
        }
    }
}

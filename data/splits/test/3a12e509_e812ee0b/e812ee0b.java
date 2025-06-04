import java.util.*;
import java.util.concurrent.Exchanger;

public class Codeforces {

    static int[] mass = new int[200_001];


    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        StringBuilder stringBuilder = new StringBuilder();
        int t = scanner.nextInt();
        for (int z =0 ; z < t ; ++z) {
            solve(scanner, stringBuilder);
        }
        System.out.print(stringBuilder);
    }

    private static void solve(Scanner scanner, StringBuilder stringBuilder) {
        int n = scanner.nextInt();
        for (int i = 0 ; i < n ; ++i) {
            mass[i] = scanner.nextInt();
        }
        String colors = scanner.next();
        List<Integer> red = new ArrayList<>(n);
        List<Integer> blue = new ArrayList<>(n);
        for (int i = 0 ; i < n ; ++i) {
            if(colors.charAt(i) == 'R') red.add(mass[i]);
            else blue.add(mass[i]);
        }
        red.sort(Comparator.reverseOrder());
        blue.sort(Comparator.naturalOrder());
        int creatd = 0;
        int upper = n;
        int down = 1;
        for (Integer integer : red) {
            if (integer > upper) {
                stringBuilder.append("NO\n");
                return;
            } else {
                creatd++;
                upper--;
            }
        }
        for (Integer integer : blue) {
            if (integer < down) {
                stringBuilder.append("NO\n");
                return;
            } else {
                creatd++;
                down++;
            }
        }
        if (creatd == n) stringBuilder.append("YES\n");
        else stringBuilder.append("NO\n");
    }
}
